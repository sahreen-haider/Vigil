import os 
from deepface import DeepFace


class Recognise_verify:
    def record_faces(self):
        if self.detected_face:
            file_path = "/Volumes/Extended/projects/Vigil/dataset/recorded_data/recorded_sample"+str(counter)+".jpg"
            cv2.imwrite(file_path, self.frame)
            counter += 1
            



    def verify_faces(self):
        models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
        # model = DeepFace.build(models)
        if self.detected_face:
            file_path = "generated_data/captured_sample.jpg"
            cv2.imwrite(file_path, self.frame)
            distance_score = DeepFace.find(img_path = file_path, db_path = "dataset" , detector_backend = "ssd")
            print(distance_score[0]["identity"])
            
        return


obj = Detect_verify()
obj.capture_live_faces()
obj.record_faces()
# obj.verify_faces()
