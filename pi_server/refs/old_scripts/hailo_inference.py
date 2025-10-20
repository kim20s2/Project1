import cv2
import numpy as np
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType)

# -------------------------------------------------------------------
# 1. GStreamer ?? (???? ?? ??)
# -------------------------------------------------------------------
# videoscale ??? ???? GStreamer ????? ??? ???? ??
gst_str = ("v4l2src device=/dev/video0 ! "
           "video/x-raw,width=640,height=480,framerate=30/1 ! "
           "videoscale ! "
           "video/x-raw,width=224,height=224 ! "
           "videoflip method=vertical-flip ! "
           "videoconvert ! "
           # ? ??? ???? ?? ?? ??? ?????.
           "video/x-raw,format=BGR ! "
           "appsink")

cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("??: GStreamer ?????? ? ? ????.")
    exit()

print("GStreamer ???? ????? ?????! 'q'? ??? ?????.")

# -------------------------------------------------------------------
# 2. Hailo ???? ? ?? ???
# -------------------------------------------------------------------
hef_path = 'resmasking.hef'
hef = HEF(hef_path)

with VDevice() as target:
    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]

    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]
    
    # ??? ?? shape? ???? ?? ?? 224x224? ?? (?????? ??)
    height, width, channels = 224, 224, 3
    
    input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    # -------------------------------------------------------------------
    # 3. ???? ??? ? ??? ??
    # -------------------------------------------------------------------
    with network_group.activate() as activated_network_group:
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("??: ???? ?? ? ????.")
                    break

                # GStreamer?? ?? 224x224? ????? 'frame'? ??
                # ??? cv2.resize() ??? ?? ??

                # --- A. ??? (???? ??) ---
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                normalized_frame = rgb_frame.astype(np.float32) / 255.0
                input_tensor = np.expand_dims(normalized_frame, axis=0)

                # --- B. Hailo ?? ---
                input_name = input_vstream_info.name
                input_data = {input_name: input_tensor}
                results_dict = infer_pipeline.infer(input_data)

                # --- C. ??? ? ?? ?? ---
                output_name = output_vstream_info.name
                output_data = results_dict[output_name]

                # 1. ?? ??? ??? ??
                emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                
                # 2. ??? ??(logits)?? ?? ?? ?? ??? ??
                # output_data? shape? (1, 7)???, ? ?? ?? [0]? ??
                logits = output_data[0]
                predicted_index = np.argmax(logits)
                
                # 3. ???? ??? ?? ??? ????
                predicted_emotion = emotion_labels[predicted_index]
                
                # (?? ??) ???(??) ??: Softmax ?? ??
                probabilities = np.exp(logits) / np.sum(np.exp(logits))
                confidence = probabilities[predicted_index]

                # 4. ?? ??
                result_text = f"Emotion: {predicted_emotion} ({confidence:.2f})"
                print(result_text)

                # 5. OpenCV? ??? ?? ???? ?? ??? ??
                # ???? ??? ??? ??? ??? ??
                cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # ?? ??? ????? ??? ??? ??
                cv2.imshow("Original with Emotion", frame)
                

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()