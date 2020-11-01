const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById('webcamButton');
const type_of_model='YOLO';
//const type_of_model='COCO';

const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0)
const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0)

var vidWidth = 0;
var vidHeight = 0;

var xStart = 0;
var yStart = 0;

// Check if webcam access is supported.
function getUserMediaSupported() {
    return !!(navigator.mediaDevices &&
        navigator.mediaDevices.getUserMedia);
}

// If webcam supported, add event listener to button for when user
// wants to activate it to call enableCam function which we will
// define in the next step.
if (getUserMediaSupported()) {
    enableWebcamButton.addEventListener('click', enableCam);
} else {
    console.warn('getUserMedia() is not supported by your browser');
}


var children = [];


//Perform prediction based on webcam using Layer model model:
function predictWebcamTF() {
    // Now let's start classifying a frame in the stream.
    detectTFMOBILE(video).then(function () {
        // Call this function again to keep predicting when the browser is ready.
        window.requestAnimationFrame(predictWebcamTF);
    });
}


// Store the resulting model in the global scope of our app.
var model = undefined;
tensorLoadModel();




// Enable the live webcam view and start classification.
function enableCam(event) {
    // Only continue if the COCO-SSD has finished loading.
    if (!model) {
        return;
    }
    // Hide the button once clicked.
    enableWebcamButton.classList.add('removed');

    // getUsermedia parameters to force video but not audio.
    const constraints = {
        video: true
    };

    // Stream video from VAR (for safari also)
    navigator.mediaDevices.getUserMedia({
        video: {
            facingMode: "environment"
        },
    }).then(stream => {
        let $video = document.querySelector('video');
        $video.srcObject = stream;
        $video.onloadedmetadata = () => {
            vidWidth = $video.videoHeight;
            vidHeight = $video.videoWidth;
            //The start position of the video (from top left corner of the viewport)
            xStart = Math.floor((vw - vidWidth) / 2);
            yStart = (Math.floor((vh - vidHeight) / 2)>=0) ? (Math.floor((vh - vidHeight) / 2)):0;
            $video.play();
            if(type_of_model=='YOLO')
            {
                $video.addEventListener('loadeddata', predictWebcamTF);
            }else{
                $video.addEventListener('loadeddata', predictWebcam);
            }
        }
    });
}




/*
doCORSRequest({
    method: 'GET',
    url: 'https://hub.tensorflow.google.cn/tensorflow/ssd_mobilenet_v2/2',
    data: null
}, function printResult(result) {
    model = result;
    //console.log(result);
    // Show demo section now model is ready to use.
    enableWebcamButton.classList.remove('invisible');
    enableWebcamButton.innerHTML = 'Start camera';
});

*/


/*----------------------------------
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>



<body>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
<!-- Load the coco-ssd model. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd"></script>


<script src="detect.js" type="text/babel"></script>

<img id="img" src="https://images.pexels.com/photos/20787/pexels-photo.jpg?auto=compress&cs=tinysrgb&dpr=1&w=500" crossorigin="anonymous"/>

<script>
    // Notice there is no 'import' statement. 'cocoSsd' and 'tf' is
    // available on the index-page because of the script tag above.

    const img = document.getElementById('img');

    // Load the model.
    //Note: cocoSsd.load() will also work on this without parameter. It will default to the coco ssd model
    //cocoSsd.load({ modelUrl: 'PATH TO MODEL JSON FILE' }).then(model => {
    cocoSsd.load().then(model => {
        // detect objects in the image.
        model.detect(img).then(predictions => {
            console.log('Predictions: ', predictions);
        });
    });
</script>

</body>


</html>


//YOLO data set:
export default [
    'person',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'sofa',
    'pottedplant',
    'bed',
    'diningtable',
    'toilet',
    'tvmonitor',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
];



 */
