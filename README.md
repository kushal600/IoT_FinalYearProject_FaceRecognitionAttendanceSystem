# Face Recognition Attendance System

## Description
The Face Recognition Attendance System is a web-based application designed to automate attendance marking using advanced face recognition technology. This project combines computer vision and real-time video processing to identify individuals and mark their attendance seamlessly.

## Features
- **Training on Images**: Train the system on provided images to recognize individual faces.
- **Real-time Video Input**: Capture video directly from a webpage using `aiortc`.
- **Frame Extraction and Processing**: Extract frames from the uploaded video for face recognition.
- **Real-time Results**: Return bounding boxes, IDs, and names of recognized individuals to the webpage.
- **Attendance Export**: Generate an Excel sheet marking recognized students as present and others as absent. The sheet is automatically downloaded to the device where the webpage is opened.

## Technologies Used
- **Python**: Backend logic and processing.
- **OpenCV (`cv2`)**: Image and video processing.
- **aiortc**: WebRTC integration for real-time video handling.
- **Excel Export**: Automated attendance logging.

## Setup Instructions

### Prerequisites
- Python 3.x
- Virtual environment (optional but recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/kushal600/IoT_FinalYearProject_FaceRecognitionAttendanceSystem.git
   cd aiortc
   cd examples/server
   
   ```
2. Create and activate a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```



### Running the Application
1. Start the backend server:
   ```bash
   python server.py
   ```
2. Open the webpage in your browser (default: [http://localhost:5000](http://localhost:5000)).
3. Upload a video for processing.

## Usage
1. **Upload Video**: Use the webpage to upload a video.
2. **Processing**: The system extracts frames and performs face recognition.
3. **Real-time Results**: View bounding boxes, IDs, and names directly on the webpage.
4. **Download Attendance**: An Excel sheet is automatically generated and downloaded, marking recognized students as present and others as absent.

## Contributing
Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature-name'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.



## Acknowledgments
- OpenCV and aiortc communities for their excellent libraries.
- Inspiration from various face recognition projects.

---

## Author

- **Kushal Shahpatel**  
- LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/shahpatel-kushal-4a4a901b6/)  
- GitHub: [GitHub Profile](https://github.com/kushal600)

---

