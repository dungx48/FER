from fer import FER
import cv2
import tensorflow as tf


def main():
    vid = cv2.VideoCapture(0)
    detector = FER(mtcnn=True)
    while (True):

        ret, frame = vid.read()
        result = detector.detect_emotions(frame)
        if len(result) != 0:
            for i in range(len(result)):

                bounding_box = result[i]["box"]
                emotions = result[i]["emotions"]
                max_score = 0
                for emotion_name, score in emotions.items():
                    if score >= max_score:
                        emotion_top = emotion_name
                        max_score = score

                cv2.rectangle(frame, (
                    bounding_box[0], bounding_box[1]),
                              (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                              (0, 155, 255), 2, )
                sring = str(emotion_top) + " " + str(max_score)
                cv2.putText(frame, sring, (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
