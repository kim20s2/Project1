from hailo_platform import HEF

hef = HEF("resmasking.hef")
input_info = hef.get_input_vstream_infos()[0]

print(f"입력 이름: {input_info.name}")
print(f"입력 형태: {input_info.shape}")  # (height, width, channels)
print(f"입력 포맷: {input_info.format}")
print(f"데이터 타입: {input_info.dtype}")
