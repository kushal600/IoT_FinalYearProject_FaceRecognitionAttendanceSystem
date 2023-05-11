from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
import cv2

video_path = "C:/Users/prath/Desktop/aiortc/examples/server/outputresult/mygeneratedvideo.avi"
temp = cv2.imread(video_path)
video = MediaStreamTrack(temp)
