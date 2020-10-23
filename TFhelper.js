var filterBoxesThreshold=0.01;
//const imageSize = 416; //TBD @@ is fixed!! need to change
const imageSize = 512; //TBD @@ is fixed!! need to change
var    IOUThreshold = 0.4;
var classProbThreshold = 0.4;






//Loading custom pretrained cocc mobilenet_v2:
//tensorflowjs_converter --input_format=tf_saved_model --output_node_names='detection_boxes,detection_classes,detection_features,detection_multiclass_scores,detection_scores,num_detections,raw_detection_boxes,raw_detection_scores' --saved_model_tags=serve --output_format=tfjs_graph_model /content/fine_tuned_model/saved_model /content/fine_tuned_model/web_model



//Image detects object that matches the preset:
async function detectTFMOBILE(imgToPredict) {
    //await this.ready:
    await tf.nextFrame();

    const tfImg = tf.browser.fromPixels(imgToPredict); //512
    const smallImg = tf.image.resizeBilinear(tfImg, [730, 610]) // 600, 450 [320, 320]
    const resized = tf.cast(smallImg, 'float32');
    const tf4d = tf.tensor4d(Array.from(resized.dataSync()), [1, 730, 610, 3]); // 600, 450

    const resized_2 = tf.cast(smallImg, 'int32');
    //var tf4d_2_ = tf.tensor4d(Array.from(resized_2.dataSync()), [1, 512, 512, 3]); // 600, 450
    var tf4d_2_ = tf.tensor4d(Array.from(resized_2.dataSync()), [1, 730, 610, 3]); // 600, 450
    const tf4d_2 = tf.cast(tf4d_2_, 'int32');
    //let predictions = await model.executeAsync({ image_tensor: tf4d }, ['detection_boxes', 'num_detections', 'detection_classes', 'detection_scores'])
    //let predictions = await model.executeAsync(tf4d_2,['detection_boxes', 'num_detections', 'detection_classes', 'detection_scores'] );

    //'detection_boxes,detection_classes,detection_features,detection_multiclass_scores,detection_scores,
    // num_detections,raw_detection_boxes,raw_detection_scores'
    let predictions = await model.executeAsync(tf4d_2);//works
    //let predictions = await model.executeAsync(tf4d_2,['detection_anchor_indices', 'detection_boxes', 'detection_scores']);//works
    let stam=1;
    //boxes,classes, scores]
    renderPredictionBoxes(predictions[4].dataSync(), predictions[1].dataSync(), predictions[2].dataSync());
    //Output:
    //[0]detection_anchor_indices [1,100]
    //[1]detection_boxes [1,100,4]
    /*
    outputs['detection_anchor_indices'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 100)
      name: StatefulPartitionedCall:0
  outputs['detection_boxes'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 100, 4)
      name: StatefulPartitionedCall:1
  outputs['detection_classes'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 100)
      name: StatefulPartitionedCall:2
  outputs['detection_multiclass_scores'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 100, 2)
      name: StatefulPartitionedCall:3
  outputs['detection_scores'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 100)
      name: StatefulPartitionedCall:4
  outputs['num_detections'] tensor_info:
      dtype: DT_FLOAT
      shape: (1)
      name: StatefulPartitionedCall:5
  outputs['raw_detection_boxes'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 1917, 4)
      name: StatefulPartitionedCall:6
  outputs['raw_detection_scores'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 1917, 2)
      name: StatefulPartitionedCall:7
     */


    /*
    let results  = model.executeAsync(tf4d_2).then(function (result) {
        const scores = result[0].dataSync();
        const boxes = result[1].dataSync();
        let stam=9;
    });
*/

/*
    const tfImg = tf.browser.fromPixels(this.$refs.video)
    const smallImg = tf.image.resizeBilinear(tfImg, [300, 300]) // 600, 450
    const resized = tf.cast(smallImg, 'float32')
    const tf4d = tf.tensor4d(Array.from(resized.dataSync()), [1, 300, 300, 3]) // 600, 450
    let predictions = await this.model.executeAsync({ image_tensor: tf4d }, ['detection_boxes', 'num_detections', 'detection_classes', 'detection_scores'])
*/
        //['detection_anchor_indices'] - additional output
    //renderPredictionBoxes(predictions[0].dataSync(), predictions[1].dataSync(), predictions[2].dataSync(), predictions[3].dataSync());
    //renderPredictionBoxes(predictions[0].dataSync(), predictions[1].dataSync(), predictions[2].dataSync());


    tfImg.dispose();
    smallImg.dispose();
    resized.dispose();
    tf4d.dispose();

}




//Rednder boxes around the detections:
//function renderPredictionBoxes (predictionBoxes, totalPredictions, predictionClasses, predictionScores)
function renderPredictionBoxes (predictionBoxes, predictionClasses, predictionScores)
{
    // get the context of canvas
    //liveView

    //Remove all detections:
    for (let i = 0; i < children.length; i++) {
        liveView.removeChild(children[i]);
    }
    children.splice(0);


    // Now lets loop through predictions and draw them to the live view if
    // they have a high confidence score.
    //for (let i = 0; i < totalPredictions[0]; i++) {
    for (let i = 0; i < 99; i++) {

        //If we are over 66% sure we are sure we classified it right, draw it!
        const minY = predictionBoxes[i * 4] * 730*100;
        const minX = predictionBoxes[i * 4 + 1] * 610*100;
        const maxY = predictionBoxes[i * 4 + 2] * 730*100;
        const maxX = predictionBoxes[i * 4 + 3] * 610*100;
        const score = predictionScores[i * 3] * 100;

        //If confidence is above 75%
        if (score > 90 && score < 100){//75) {
            const p = document.createElement('p');
            p.innerText = Math.round(score) + '% ' + 'MNM';
            p.style = 'margin-left: ' + (minX-10) + 'px; ' +
                'margin-top: ' + (maxY-10) + 'px; ' +
                'width: ' + (maxX-minX) + 'px; ' +
                'top: 0; ' +
                'left: 0;';
            //p.style = 'position: absolute'; //KOSTA
            const highlighter = document.createElement('div');
            highlighter.setAttribute('class', 'highlighter');
            highlighter.style = 'left: ' + minX + 'px; ' +
                'top: ' + maxY + 'px; ' +
                'width: ' + (maxX-minX) + 'px; ' +
                'height: ' + (maxY-minY) + 'px;';

            liveView.appendChild(highlighter);
            highlighter.appendChild(p);
            children.push(highlighter);
        }
    }


/*
    const ctx = this.$refs.canvas.getContext('2d')
    // clear the canvas
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    // draw results
    for (let i = 0; i < totalPredictions[0]; i++) {
        const minY = predictionBoxes[i * 4] * 450
        const minX = predictionBoxes[i * 4 + 1] * 600
        const maxY = predictionBoxes[i * 4 + 2] * 450
        const maxX = predictionBoxes[i * 4 + 3] * 600
        const score = predictionScores[i * 3] * 100
        if (score > 75) {
            ctx.beginPath()
            ctx.rect(minX, minY, maxX - minX, maxY - minY)
            ctx.lineWidth = 3
            ctx.strokeStyle = 'red'
            ctx.fillStyle = 'red'
            ctx.stroke()
            ctx.shadowColor = 'white'
            ctx.shadowBlur = 10
            ctx.font = '14px Arial bold'
            ctx.fillText(
                `${score.toFixed(1)} - Jagermeister bottle`,
                minX,
                minY > 10 ? minY - 5 : 10
            )
        }
    }
    */
}















//Used for post processing for YOLO
async function filterBoxes(
    boxes,
    boxConfidence,
    boxClassProbs,
    threshold,
) {
    const boxScores = tf.mul(boxConfidence, boxClassProbs);
    const boxClasses = tf.argMax(boxScores, -1);
    const boxClassScores = tf.max(boxScores, -1);

    const predictionMask = tf.greaterEqual(boxClassScores, tf.scalar(threshold));

    const maskArr = await predictionMask.data();

    const indicesArr = [];
    for (let i = 0; i < maskArr.length; i += 1) {
        const v = maskArr[i];
        if (v) {
            indicesArr.push(i);
        }
    }

    if (indicesArr.length === 0) {
        return [null, null, null];
    }

    const indices = tf.tensor1d(indicesArr, 'int32');

    return [
        tf.gather(boxes.reshape([maskArr.length, 4]), indices),
        tf.gather(boxClassScores.flatten(), indices),
        tf.gather(boxClasses.flatten(), indices),
    ];
}

//Used as post processing for YOLO
const boxesToCorners = (boxXY, boxWH) => {
    const two = tf.tensor1d([2.0]);
    const boxMins = tf.sub(boxXY, tf.div(boxWH, two));
    const boxMaxes = tf.add(boxXY, tf.div(boxWH, two));

    const dim0 = boxMins.shape[0];
    const dim1 = boxMins.shape[1];
    const dim2 = boxMins.shape[2];
    const size = [dim0, dim1, dim2, 1];

    return tf.concat([
        boxMins.slice([0, 0, 0, 1], size),
        boxMins.slice([0, 0, 0, 0], size),
        boxMaxes.slice([0, 0, 0, 1], size),
        boxMaxes.slice([0, 0, 0, 0], size),
    ], 3);
};

// Convert yolo output to bounding box + prob tensors
/* eslint no-param-reassign: 0 */
function head(feats, anchors, numClasses) {
    const numAnchors = anchors.shape[0];

    const anchorsTensor = tf.reshape(anchors, [1, 1, numAnchors, 2]);

    let convDims = feats.shape.slice(1, 3);

    // For later use
    const convDims0 = convDims[0];
    const convDims1 = convDims[1];

    let convHeightIndex = tf.range(0, convDims[0]);
    let convWidthIndex = tf.range(0, convDims[1]);
    convHeightIndex = tf.tile(convHeightIndex, [convDims[1]]);

    convWidthIndex = tf.tile(tf.expandDims(convWidthIndex, 0), [convDims[0], 1]);
    convWidthIndex = tf.transpose(convWidthIndex).flatten();

    let convIndex = tf.transpose(tf.stack([convHeightIndex, convWidthIndex]));
    convIndex = tf.reshape(convIndex, [convDims[0], convDims[1], 1, 2]);
    convIndex = tf.cast(convIndex, feats.dtype);

    feats = tf.reshape(feats, [convDims[0], convDims[1], numAnchors, numClasses + 5]);
    convDims = tf.cast(tf.reshape(tf.tensor1d(convDims), [1, 1, 1, 2]), feats.dtype);

    let boxXY = tf.sigmoid(feats.slice([0, 0, 0, 0], [convDims0, convDims1, numAnchors, 2]));
    let boxWH = tf.exp(feats.slice([0, 0, 0, 2], [convDims0, convDims1, numAnchors, 2]));
    const boxConfidence = tf.sigmoid(feats.slice([0, 0, 0, 4], [convDims0, convDims1, numAnchors, 1]));
    const boxClassProbs = tf.softmax(feats.slice([0, 0, 0, 5], [convDims0, convDims1, numAnchors, numClasses]));

    boxXY = tf.div(tf.add(boxXY, convIndex), convDims);
    boxWH = tf.div(tf.mul(boxWH, anchorsTensor), convDims);

    return [boxXY, boxWH, boxConfidence, boxClassProbs];
}

//Post processing for YOLO:
async function filterBoxes(
    boxes,
    boxConfidence,
    boxClassProbs,
    threshold,
) {
    const boxScores = tf.mul(boxConfidence, boxClassProbs);
    const boxClasses = tf.argMax(boxScores, -1);
    const boxClassScores = tf.max(boxScores, -1);

    const predictionMask = tf.greaterEqual(boxClassScores, tf.scalar(threshold));

    const maskArr = await predictionMask.data();

    const indicesArr = [];
    for (let i = 0; i < maskArr.length; i += 1) {
        const v = maskArr[i];
        if (v) {
            indicesArr.push(i);
        }
    }

    if (indicesArr.length === 0) {
        return [null, null, null];
    }

    const indices = tf.tensor1d(indicesArr, 'int32');

    return [
        tf.gather(boxes.reshape([maskArr.length, 4]), indices),
        tf.gather(boxClassScores.flatten(), indices),
        tf.gather(boxClasses.flatten(), indices),
    ];
}

//Post processing for YOLO:
const boxIOU = (a, b) => boxIntersection(a, b) / boxUnion(a, b);
const boxIntersection = (a, b) => {
    const w = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
    const h = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
    if (w < 0 || h < 0) {
        return 0;
    }
    return w * h;
};
const boxUnion = (a, b) => {
    const i = boxIntersection(a, b);
    return (((a[3] - a[1]) * (a[2] - a[0])) + ((b[3] - b[1]) * (b[2] - b[0]))) - i;
};


//Post processing for YOLO
const nonMaxSuppression = (boxes, scores, iouThreshold) => {
    // Zip together scores, box corners, and index
    const zipped = [];
    for (let i = 0; i < scores.length; i += 1) {
        zipped.push([
            scores[i], [boxes[4 * i], boxes[(4 * i) + 1], boxes[(4 * i) + 2], boxes[(4 * i) + 3]], i,
        ]);
    }
    const sortedBoxes = zipped.sort((a, b) => b[0] - a[0]);
    const selectedBoxes = [];

    sortedBoxes.forEach((box) => {
        let add = true;
        for (let i = 0; i < selectedBoxes.length; i += 1) {
            const curIOU = boxIOU(box[1], selectedBoxes[i][1]);
            if (curIOU > iouThreshold) {
                add = false;
                break;
            }
        }
        if (add) {
            selectedBoxes.push(box);
        }
    });

    return [
        selectedBoxes.map(e => e[2]),
        selectedBoxes.map(e => e[1]),
        selectedBoxes.map(e => e[0]),
    ];
};




// Static Method: crop the image
const cropImage = (img) => {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
};

// Static Method: image to tf tensor
function imgToTensor(input, size = null) {
    return tf.tidy(() => {
        let img = tf.browser.fromPixels(input);
        if (size) {
            img = tf.image.resizeBilinear(img, size);
        }
        //const croppedImage = cropImage(img);
        const croppedImage = img;//KOSTA
        //const batchedImage = croppedImage.expandDims(0);
        const batchedImage = croppedImage.expandDims(0);
        //return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
        //return batchedImage.div(tf.scalar(127)).sub(tf.scalar(1));
        //tf.dtypes.cast(x, tf.int32)
        //return (tf.cast(batchedImage.div(tf.scalar(127)).sub(tf.scalar(1)),'int32')); //KOSTA_CHANGE
        //return batchedImage.to_int32().div(tf.scalar(127)).sub(tf.scalar(1)); //KOSTA_CHANGE
        return (tf.cast(batchedImage.div(tf.scalar(127)).sub(tf.scalar(1)),'int32')); //KOSTA_CHANGE
    });
}

//Post processing for YOLO:
const ANCHORS = tf.tensor2d([
    [0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434],
    [7.88282, 3.52778], [9.77052, 9.16828],
]);

//Image detects object that matches the preset:
async function detectTF(imgToPredict) {

    //await this.ready; ??
    await tf.nextFrame();

    /*
    //this.isPredicting = true; ??
    const [allBoxes, boxConfidence, boxClassProbs] = tf.tidy(async () => {
        //const input = imgToTensor(imgToPredict, [imageSize, imageSize]);
        const input = imgToTensor(imgToPredict, [512, 512]);
        const activation = await model.executeAsync(input);//model.predict(input);//model.predict(input);
        const [boxXY, boxWH, bConfidence, bClassProbs] = head(activation, ANCHORS, 80);
        const aBoxes = boxesToCorners(boxXY, boxWH);
        return [aBoxes, bConfidence, bClassProbs];
    });
     */

    //const input = imgToTensor(imgToPredict, [imageSize, imageSize]);
    //const input = imgToTensor(imgToPredict, [512, 512]);
    const input = imgToTensor(imgToPredict, [224,224]);//[1200, 1600]);
    const activation = await model.executeAsync(input);//model.predict(input);//model.predict(input);
    //const activation =  model.predict(input);
    const [boxXY, boxWH, boxConfidence, boxClassProbs] = head(activation, ANCHORS, 80);
    const allBoxes = boxesToCorners(boxXY, boxWH);
    tf.tidy();


    const [boxes, scores, classes] = await filterBoxes(allBoxes, boxConfidence, boxClassProbs, filterBoxesThreshold);

    // If all boxes have been filtered out
    if (boxes == null) {
        return [];
    }

    const width = tf.scalar(imageSize);
    const height = tf.scalar(imageSize);
    const imageDims = tf.stack([height, width, height, width]).reshape([1, 4]);
    const boxesModified = tf.mul(boxes, imageDims);

    const [preKeepBoxesArr, scoresArr] = await Promise.all([
        boxesModified.data(), scores.data(),
    ]);

    const [keepIndx, boxesArr, keepScores] = nonMaxSuppression(
        preKeepBoxesArr,
        scoresArr,
        IOUThreshold,
    );

    const classesIndxArr = await classes.gather(tf.tensor1d(keepIndx, 'int32')).data();

    const results = [];

    classesIndxArr.forEach((classIndx, i) => {
        const classProb = keepScores[i];
        if (classProb < classProbThreshold) {
            return;
        }

        const className = CLASS_NAMES[classIndx];
        let [y, x, h, w] = boxesArr[i];

        y = Math.max(0, y);
        x = Math.max(0, x);
        h = Math.min(imageSize, h) - y;
        w = Math.min(imageSize, w) - x;

        //x,y is low left corner:
        const resultObj = {
            label: className,
            confidence: classProb,
            x: x / imageSize,
            y: y / imageSize,
            w: w / imageSize,
            h: h / imageSize,
        };

        results.push(resultObj);
    });

    //this.isPredicting = false; ??
    return results;
}



//From ml5 loader:--------------------------------------------------
async function asyncLoadModel(model_url) {

    //let modelUrl = `${model_url}/model.json`;
    //model - defined in scirpt.js
    //model = await tf.loadLayersModel(model_url);
    model = await tf.loadGraphModel(model_url); //kind of works in 22/10/20
    //model = await tf.loadLayersModel(model_url);//based on https://github.com/tensorflow/tfjs/blob/master/tfjs-converter/README.md#4472

    console.log('Model loaded');
    //tf.env().set('WEBGPU_CPU_FORWARD', false);//KOSTA: work around slice issue in models.
    //tf.env().set('WEBGL_CPU_FORWARD', false);
    //defined in script.js:
    enableWebcamButton.classList.remove('invisible');
    enableWebcamButton.innerHTML = 'Start camera';
}


//Loading tensor.js models - exported function
function tensorLoadModel()
{
    //model_name_url = "https://hub.tensorflow.google.cn/tensorflow/ssd_mobilenet_v2/2";
    //let model_name_url = "https://raw.githubusercontent.com/ml5js/ml5-data-and-training/master/models/YOLO/model.json";
    //https://raw.githubusercontent.com/KostaMalsev/ImageRecognition/master/model.json
    //let model_name_url = "https://raw.githubusercontent.com/KostaMalsev/ImageRecognition/master/model.json";
    //let model_name_url = "https://raw.githubusercontent.com/KostaMalsev/ImageRecognition/master/model/model.json";
    //let model_name_url = "https://raw.githubusercontent.com/KostaMalsev/ImageRecognition/master/model/mobile_netv2/webmodel/model.json";
    let model_name_url = "https://raw.githubusercontent.com/KostaMalsev/ImageRecognition/master/model/mobile_netv2/web_model2/model.json";
    //https://github.com/KostaMalsev/ImageRecognition/tree/master/model/mobile_netv2/web_model2
    asyncLoadModel(model_name_url);
}






var mobilenet;
//const mobilenetDemo = async () => {
function loadModelMobilenet() {
    //mobilenet = await tf.loadLayersModel("https://hub.tensorflow.google.cn/tensorflow/ssd_mobilenet_v2/2");
    //mobilenet = await tf.loadLayersModel("https://hub.tensorflow.google.cn/tensorflow/ssd_mobilenet_v2/2");
    mobilenet = ml5.objectDetector('MobileNet', () => {
        //Using ml5 library:
        model = mobilenet;
        // Show demo section now model is ready to use.
        enableWebcamButton.classList.remove('invisible');
        enableWebcamButton.innerHTML = 'Start camera';
    });
    //"https://hub.tensorflow.google.cn/tensorflow/ssd_mobilenet_v2/2"
    //https://hub.tensorflow.google.cn/tensorflow/ssd_mobilenet_v2/2"
}

//const mobilenetDemo = async () => {
function loadModelYOLO() {
    //mobilenet = await tf.loadLayersModel("https://hub.tensorflow.google.cn/tensorflow/ssd_mobilenet_v2/2");
    //mobilenet = await tf.loadLayersModel("https://hub.tensorflow.google.cn/tensorflow/ssd_mobilenet_v2/2");
    //mobilenet =  ml5.objectDetector('MobileNet', ()=>{
    //Using ml5 library:
    mobilenet =  ml5.YOLO(video,()=>{
        model = mobilenet;
        // Show demo section now model is ready to use.
        enableWebcamButton.classList.remove('invisible');
        enableWebcamButton.innerHTML = 'Start camera';
    });
    //"https://hub.tensorflow.google.cn/tensorflow/ssd_mobilenet_v2/2"
    //https://hub.tensorflow.google.cn/tensorflow/ssd_mobilenet_v2/2"
}



//Utility to overcome CORS limitations in GET and POST:
var cors_api_url = 'https://cors-anywhere.herokuapp.com/';
function doCORSRequest(options, printResult) {
    var x = new XMLHttpRequest();
    x.open(options.method, cors_api_url + options.url);
    x.onload = x.onerror = function() {
        printResult(
            (x.responseText || '')
        );
    };
    if (/^POST/i.test(options.method)) {
        x.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
    }
    x.send(options.data);
}


/*
function isAbsoluteURL(str) {
    const pattern = new RegExp('^(?:[a-z]+:)?//', 'i');
    return !!pattern.test(str);
}

function getModelPath(absoluteOrRelativeUrl) {
    const modelJsonPath = isAbsoluteURL(absoluteOrRelativeUrl) ? absoluteOrRelativeUrl : window.location.pathname + absoluteOrRelativeUrl
    return modelJsonPath;
}
*/


//From ml5 loader:--------------------------------------------------

/*
if (this.videoElt && !this.video) {
    this.video = await this.loadVideo();
}

if(modelLoader.isAbsoluteURL(this.modelUrl) === true){
    //tf is declared in tensor flow library
    this.model = await tf.loadLayersModel(this.modelUrl);
} else {
    const modelPath = modelLoader.getModelPath(this.modelUrl);
    this.modelUrl = `${modelPath}/model.json`;
    this.model = await tf.loadLayersModel(this.modelUrl);
}

this.modelReady = true;
return this;
 */