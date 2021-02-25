//Store hooks and video sizes:
const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById('webcamButton');
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

// If webcam supported, add event listener to activation button:
if (getUserMediaSupported()) {
    enableWebcamButton.addEventListener('click', enableCam);
} else {
    console.warn('getUserMedia() is not supported by your browser');
}

// Enable the live webcam view and start classification.
function enableCam(event) {
  // Only continue if model has finished loading.
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
      //Attach detection model to loaded data event:
      $video.addEventListener('loadeddata', predictWebcamTF);
    }
  });
}



var model = undefined;
model_url = 'https://raw.githubusercontent.com/KostaMalsev/ImageRecognition/master/model/mobile_netv2/web_model2/model.json';
//Call load function
asyncLoadModel(model_url);

//Function Loads the GraphModel type model of
async function asyncLoadModel(model_url) {
    model = await tf.loadGraphModel(model_url);
    console.log('Model loaded');
    //Enable start button:
    enableWebcamButton.classList.remove('invisible');
    enableWebcamButton.innerHTML = 'Start camera';
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
const imageSize = 512;
//Match prob. threshold for object detection:
var classProbThreshold = 0.4;//40%
//Image detects object that matches the preset:
async function detectTFMOBILE(imgToPredict) {

    //Get next video frame:
    await tf.nextFrame();
    //Create tensor from image:
    const tfImg = tf.browser.fromPixels(imgToPredict);

    //Create smaller image which fits the detection size
    const smallImg = tf.image.resizeBilinear(tfImg, [vidHeight,vidWidth]);


    const resized = tf.cast(smallImg, 'int32');
    var tf4d_ = tf.tensor4d(Array.from(resized.dataSync()), [1,vidHeight, vidWidth, 3]);
    const tf4d = tf.cast(tf4d_, 'int32');

    //Perform the detection with your layer model:
    let predictions = await model.executeAsync(tf4d);
    
    //Draw box around the detected object:
    renderPredictionBoxes(predictions[4].dataSync(), predictions[1].dataSync(), predictions[2].dataSync());
    //Dispose of the tensors (so it won't consume memory)
    tfImg.dispose();
    smallImg.dispose();
    resized.dispose();
    tf4d.dispose();
    
}



//Function Renders boxes around the detections:
function renderPredictionBoxes (predictionBoxes, predictionClasses, predictionScores)
{
    //Remove all detections:
    for (let i = 0; i < children.length; i++) {
        liveView.removeChild(children[i]);
    }
    children.splice(0);
//Loop through predictions and draw them to the live view if they have a high confidence score.
    for (let i = 0; i < 99; i++) {
//If we are over 66% sure we are sure we classified it right, draw it!
        const minY = (predictionBoxes[i * 4] * vidHeight+yStart).toFixed(0);
        const minX = (predictionBoxes[i * 4 + 1] * vidWidth+xStart).toFixed(0);
        const maxY = (predictionBoxes[i * 4 + 2] * vidHeight+yStart).toFixed(0);
        const maxX = (predictionBoxes[i * 4 + 3] * vidWidth+xStart).toFixed(0);
        const score = predictionScores[i * 3] * 100;
const width_ = (maxX-minX).toFixed(0);
        const height_ = (maxY-minY).toFixed(0);
//If confidence is above 70%
        if (score > 70 && score < 100){
            const highlighter = document.createElement('div');
            highlighter.setAttribute('class', 'highlighter');
            highlighter.style = 'left: ' + minX + 'px; ' +
                'top: ' + minY + 'px; ' +
                'width: ' + width_ + 'px; ' +
                'height: ' + height_ + 'px;';
            highlighter.innerHTML = '<p>'+Math.round(score) + '% ' + 'Your Object Name'+'</p>';
            liveView.appendChild(highlighter);
            children.push(highlighter);
        }
    }
}
