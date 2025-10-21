import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTModule:
    def __init__(self, engine_path):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        nb = self.engine.num_bindings
        self.bindings = [0] * nb

        # public attrs
        self.input_names = []
        self.output_names = []
        self.shapes = {}      # name -> tuple
        self.dtypes = {}      # name -> np.dtype
        self.host = {}        # name -> page-locked flat buffer (np array 1D)
        self.device = {}      # name -> device pointer (DeviceAllocation)

        for i in range(nb):
            name = self.engine.get_binding_name(i)
            is_input = self.engine.binding_is_input(i)

            # dtype/shape from engine
            trt_dtype = self.engine.get_binding_dtype(i)
            np_dtype = trt.nptype(trt_dtype)
            # shape: prefer context binding shape (static 엔진이면 동일)
            shape = tuple(self.context.get_binding_shape(i))

            # 동적(-1) 차원이 보이면 배치=1 대입 시도 (일반적 경우)
            if any(d < 0 for d in shape):
                # batch 차원만 -1인 경우가 대부분이므로 1로 교체
                shape = tuple(1 if d < 0 else d for d in shape)
                # 추가로 context에 바인딩 shape 지정 (가능할 때만)
                try:
                    self.context.set_binding_shape(i, shape)
                except Exception:
                    pass  # 정적 엔진이면 필요 없음

            # 버퍼 준비 (page-locked host, device)
            vol = int(trt.volume(shape))
            host_mem = cuda.pagelocked_empty(vol, np_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # 기록
            self.shapes[name] = shape
            self.dtypes[name] = np_dtype
            self.host[name] = host_mem
            self.device[name] = device_mem
            self.bindings[i] = int(device_mem)

            if is_input:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer(self, inp):
        """
        inp: np.ndarray (단일 입력 엔진) 또는 {name: np.ndarray}
        반환: 단일 출력이면 np.ndarray, 다중 출력이면 [np.ndarray, ...]
        """
        if isinstance(inp, dict):
            feed = inp
        else:
            if len(self.input_names) != 1:
                raise ValueError("Multiple inputs expected. Pass a dict {name: array}.")
            feed = {self.input_names[0]: inp}

        # ====== H2D (입력 복사) ======
        for in_name, arr in feed.items():
            if in_name not in self.shapes:
                raise KeyError(f"Unknown input name: {in_name}")

            exp = tuple(self.shapes[in_name])   # expected shape from engine
            want_dtype = self.dtypes[in_name]   # np dtype expected by engine

            arr = np.asarray(arr)
            # 레이아웃 자동 전환 (NCHW/NHWC 간 맞춰줌)
            if arr.ndim == 4 and len(exp) == 4 and arr.shape != exp:
                # NCHW -> NHWC
                if arr.shape[1] in (1, 3) and exp[-1] in (1, 3) and arr.shape[1] != exp[-1]:
                    arr = np.transpose(arr, (0, 2, 3, 1))
                # NHWC -> NCHW
                elif arr.shape[-1] in (1, 3) and exp[1] in (1, 3) and arr.shape[-1] != exp[1]:
                    arr = np.transpose(arr, (0, 3, 1, 2))

            # 필요하면 배치 차원 추가
            if arr.ndim == len(exp) - 1:
                arr = np.expand_dims(arr, 0)

            # 최종 shape 강제
            if tuple(arr.shape) != exp:
                try:
                    arr = np.reshape(arr, exp)
                except Exception as e:
                    raise ValueError(f"Input {in_name} shape mismatch: got {arr.shape}, expected {exp}") from e

            # !!! dtype을 엔진에 맞춤 (float32/int32 등) !!!
            arr = arr.astype(want_dtype, copy=False)

            np.copyto(self.host[in_name], arr.ravel())
            cuda.memcpy_htod_async(self.device[in_name], self.host[in_name], self.stream)

        # ====== 실행 ======
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)

        # ====== D2H (출력 복사) ======
        outs_flat = []
        for out_name in self.output_names:
            cuda.memcpy_dtoh_async(self.host[out_name], self.device[out_name], self.stream)
            outs_flat.append(self.host[out_name].copy())

        self.stream.synchronize()

        # 출력 shape/dtype 복원
        final_out = []
        for out_name, flat in zip(self.output_names, outs_flat):
            shp = tuple(self.shapes[out_name])
            dt = self.dtypes[out_name]
            final_out.append(flat.view(dt).reshape(shp))

        if len(final_out) == 1:
            return final_out[0]
        return final_out

    def destroy(self):
        # 명시적 자원 해제(옵션)
        for name in list(self.device.keys()):
            try:
                self.device[name].free()
            except Exception:
                pass
        self.device.clear()
        self.host.clear()
        self.bindings = []
        try:
            del self.context
        except Exception:
            pass
        try:
            del self.engine
        except Exception:
            pass
