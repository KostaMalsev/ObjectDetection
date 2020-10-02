const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById('webcamButton');
const type_of_model='YOLO';


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
//Perform prediction based on webcam using coco model:
function predictWebcam() {
    // Now let's start classifying a frame in the stream.
    model.detect(video).then(function (predictions) {
        // Remove any highlighting we did previous frame.
        for (let i = 0; i < children.length; i++) {
            liveView.removeChild(children[i]);
        }
        children.splice(0);

        // Now lets loop through predictions and draw them to the live view if
        // they have a high confidence score.
        for (let n = 0; n < predictions.length; n++) {
            // If we are over 66% sure we are sure we classified it right, draw it!
            if (predictions[n].score > 0.66) {
                const p = document.createElement('p');
                p.innerText = Math.round(parseFloat(predictions[n].score) * 100) + '% ' + predictions[n].class;
                p.style = 'margin-left: ' + predictions[n].x*video.videoWidth + 'px; ' +
                    'margin-top: ' + (predictions[n].y*video.videoHeight - 10) + 'px; ' +
                    'width: ' + (predictions[n].width*video.videoWidth - 10) + 'px; ' +
                    'top: 0; ' +
                    'left: 0;';
                //p.style = 'position: absolute'; //KOSTA
                const highlighter = document.createElement('div');
                highlighter.setAttribute('class', 'highlighter');
                highlighter.style = 'left: ' + predictions[n].x*video.videoWidth + 'px; ' +
                    'top: ' + predictions[n].y*x*video.videoHeight + 'px; ' +
                    'width: ' + predictions[n].width*video.videoWidth + 'px; ' +
                    'height: ' + predictions[n].height*video.videoHeight + 'px;';

                liveView.appendChild(highlighter);
                highlighter.appendChild(p);
                children.push(highlighter);

            }
        }

        // Call this function again to keep predicting when the browser is ready.
        window.requestAnimationFrame(predictWebcam);
    });
}

//Perform prediction based on webcam using coco model:
function predictWebcamTF() {
    // Now let's start classifying a frame in the stream.
    detectTF(video).then(function (predictions) {
        // Remove any highlighting we did previous frame.
        for (let i = 0; i < children.length; i++) {
            liveView.removeChild(children[i]);
        }
        children.splice(0);

        // Now lets loop through predictions and draw them to the live view if
        // they have a high confidence score.
        for (let n = 0; n < predictions.length; n++) {
            // If we are over 66% sure we are sure we classified it right, draw it!
            //if (predictions[n].score > 0.66) {
                const p = document.createElement('p');
                p.innerText = Math.round(parseFloat(predictions[n].confidence) * 100) + '% ' + predictions[n].label;
                p.style = 'margin-left: ' + predictions[n].x*video.videoWidth + 'px; ' +
                    'margin-top: ' + (predictions[n].y*video.videoHeight - 10) + 'px; ' +
                    'width: ' + (predictions[n].width*video.videoWidth - 10) + 'px; ' +
                    'top: 0; ' +
                    'left: 0;';
                //p.style = 'position: absolute'; //KOSTA
                const highlighter = document.createElement('div');
                highlighter.setAttribute('class', 'highlighter');
                highlighter.style = 'left: ' + predictions[n].x*video.videoWidth + 'px; ' +
                    'top: ' + predictions[n].y*video.videoHeight + 'px; ' +
                    'width: ' + predictions[n].width*video.videoWidth + 'px; ' +
                    'height: ' + predictions[n].height*video.videoHeight + 'px;';

                liveView.appendChild(highlighter);
                highlighter.appendChild(p);
                children.push(highlighter);

            //}
        }

        // Call this function again to keep predicting when the browser is ready.
        window.requestAnimationFrame(predictWebcamTF);
    });
}







// Store the resulting model in the global scope of our app.
var model = undefined;

//Loading coco ssd pretrained model:
/*
cocoSsd.load().then(function (loadedModel) {
    model = loadedModel;
    //Enable buttons:
    enableWebcamButton.classList.remove('invisible');
    enableWebcamButton.innerHTML = 'Start camera';
});
*/


//Load tensor model:
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
