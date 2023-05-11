import argparse
import asyncio
import json
import random
import logging
from PIL import Image
import os
import ssl
import uuid
import numpy as np
import faceRecognition as fr
import math
from os import listdir
import glob
import face_recognition
import pandas as pd
import csv
import shutil


import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import cv2
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()


path = 'examples\server\image'

images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print("classname")
print(classNames)
encodeListKnown = fr.findEncodings(images)
print('Encoding Complete')

dir = "outputtesting"
name = "test"
parent_dir = f'C:/Users/prath/Desktop/aiortc/examples/server/'
path = os.path.join(parent_dir, dir)
if not os.path.exists(path):
    os.makedirs(path)


students = []
id = []
frame_list = []


import time
# Video Generating function
def generate_video():
    os.chdir("C:/Users/prath/Desktop/aiortc/examples/server/outputresult")
    path = "C:/Users/prath/Desktop/aiortc/examples/server/outputresult"

    # print("first element of output:",os.listdir(path)[0])
    os.remove(path+"/"+os.listdir(path)[0])
    mean_height = 0
    mean_width = 0

    num_of_images = len(os.listdir('.'))
    # print(num_of_images)

    for file in os.listdir('.'):
        im = Image.open(os.path.join(path, file))
        width, height = im.size
        mean_width += width
        mean_height += height


    mean_width = int(mean_width / num_of_images)
    mean_height = int(mean_height / num_of_images)


    # Resizing of the images to give
    # them same width and height
    for i in range(len(os.listdir('C:/Users/prath/Desktop/aiortc/examples/server/outputresult'))):
        file = 'C:/Users/prath/Desktop/aiortc/examples/server/outputresult'+'/test'+str(i)
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"):
            # opening image using PIL Image
            im = Image.open(os.path.join(path, file))

            # im.size includes the height and width of image
            width, height = im.size

            # resizing
            imResize = im.resize((mean_width, mean_height), Image.Resampling.LANCZOS)
            imResize.save(file, 'JPEG', quality=95)  # setting quality
            # printing each resized image name


        image_folder = '.'  # make sure to use your folder
        video_name = 'mygeneratedvideo.avi'
        os.chdir("C:/Users/prath/Desktop/aiortc/examples/server/outputresult")

        images = [img for img in os.listdir(image_folder)
                if img.endswith(".jpg")
                or img.endswith(".jpeg")
                or img.endswith("png")]

        # Array images should only consider
        # the image files ignoring others if any

        frame = cv2.imread(os.path.join(image_folder, images[0]))

        # setting the frame width, height width
        # the width, height of first image
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 15, (width, height))

        # Appending the images to the video one by one
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        # Deallocating memories taken for window creation
        cv2.destroyAllWindows()
        video.release()  # releasing the video generated
    print("Video done!!")


async def outputimages():
    output_frame_list = []
    for i in (os.listdir("C:/Users/prath/Desktop/aiortc/examples/server/outputresult")):
        os.remove("C:/Users/prath/Desktop/aiortc/examples/server/outputresult/"+i)
    count = 0
    folder_dir = "C:/Users/prath/Desktop/aiortc/examples/server/outputtesting"
    k = len(os.listdir(folder_dir))
    for images in os.listdir(folder_dir):
        if (images.endswith(".jpg") and k > 0):

            img_dir = "C:/Users/prath/Desktop/aiortc/examples/server/outputtesting"  # Enter Directory of all images
            data_path = os.path.join(folder_dir, '*g')
            files = glob.glob(data_path)
            data = []

            for f1 in files:
                print(f1)
                img = cv2.imread(f1)
                imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

                facesCurFrame = face_recognition.face_locations(imgS)
                encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        students.append(name)
                        id.append(matchIndex)
                k -= 1
                file_name2 = f"examples/server/outputresult/" + "test" + str(count) + ".jpg"
                cv2.imwrite(file_name2, img)
                new_frame = VideoFrame.from_ndarray(img, format="bgr24")
                output_frame_list.append(new_frame)
                count += 1
    #until all the images in the testing folder is processed the video will not generate...
    unique_students = set(students)
    present_students = list(unique_students)
    # print("present students" , present_students)
    unique_id = set(id)
    present_id = list(unique_id)
    label = []
    no=1
    for i in unique_id:
        label.append(i+1)
    # print(l2)


    with open('C:/Users/prath/Desktop/aiortc/examples/server/student.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SN", "name", "attendance"])
        for i in classNames:
            writer.writerow([str(no), i,"absent"])
            no+=1

    df = pd.read_csv('C:/Users/prath/Desktop/aiortc/examples/server/student.csv')

    with open('C:/Users/prath/Desktop/aiortc/examples/server/student.csv', 'r') as file: 
        reader = csv.reader(file) 
        for row in reader:
            if row[0] == "SN":
                continue 
            no = int(row[0])
            
            if no in label:
                df['attendance'][no-1] = 'present'
    df.to_csv('C:/Users/prath/Desktop/aiortc/examples/server/student.csv', index=False)

    generate_video()


class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track
        self.count = 0

    async def recv(self):
        file_name = f"examples/server/outputtesting/" + "test" + str(self.count) + ".jpg"
        frame = await self.track.recv()
        self.count += 1
        img = frame.to_ndarray(format="bgr24")
        temp = cv2.imwrite(file_name, img)
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        for i in (os.listdir("C:/Users/prath/Desktop/aiortc/examples/server/outputtesting")):
            os.remove("C:/Users/prath/Desktop/aiortc/examples/server/outputtesting/"+i)
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track)
                ))
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            await outputimages()
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
