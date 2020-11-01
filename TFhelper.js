var filterBoxesThreshold=0.01;
//const imageSize = 416; //TBD @@ is fixed!! need to change
const imageSize = 512; //TBD @@ is fixed!! need to change
var    IOUThreshold = 0.4;
var classProbThreshold = 0.4;



//Image detects object that matches the preset:
async function detectTFMOBILE(imgToPredict) {
    //await this.ready:
    await tf.nextFrame();

    //Size of the video:
    //vidWidth ;
    //vidHeight ;

    //Up-Left corner position of the video:
    //xStart ;
    //yStart ;

    const tfImg = tf.browser.fromPixels(imgToPredict); //512
    const smallImg = tf.image.resizeBilinear(tfImg, [vidHeight,vidWidth]); //y:1200 x:1600 (trained image size) 600, 450 [320, 320]
    const resized = tf.cast(smallImg, 'float32');
    const tf4d = tf.tensor4d(Array.from(resized.dataSync()), [1, vidHeight, vidWidth, 3]); //730,610 600, 450

    const resized_2 = tf.cast(smallImg, 'int32');
    //var tf4d_2_ = tf.tensor4d(Array.from(resized_2.dataSync()), [1, 512, 512, 3]); // 600, 450
    var tf4d_2_ = tf.tensor4d(Array.from(resized_2.dataSync()), [1,vidHeight, vidWidth, 3]); // 600, 450
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

    tfImg.dispose();
    smallImg.dispose();
    resized.dispose();
    tf4d.dispose();
}

//Rendeder boxes around the detections:
//function renderPredictionBoxes (predictionBoxes, totalPredictions, predictionClasses, predictionScores)
function renderPredictionBoxes (predictionBoxes, predictionClasses, predictionScores)
{
    // get the context of canvas
    //liveView

    //Size of the video:
    //vidWidth ;
    //vidHeight ;
    //Up-Left corner position of the video:
    //xStart ;
    //yStart ;

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
        const minY = (predictionBoxes[i * 4] * vidHeight+yStart).toFixed(0); //730, 610
        const minX = (predictionBoxes[i * 4 + 1] * vidWidth+xStart).toFixed(0);
        const maxY = (predictionBoxes[i * 4 + 2] * vidHeight+yStart).toFixed(0);
        const maxX = (predictionBoxes[i * 4 + 3] * vidWidth+xStart).toFixed(0);
        const score = predictionScores[i * 3] * 100;

        const width_ = (maxX-minX).toFixed(0);
        const height_ = (maxY-minY).toFixed(0);

        //If confidence is above 75%
        if (score > 70 && score < 100){//75) {
            /*const p = document.createElement('p');
            p.innerText = Math.round(score) + '% ' + 'MNM';
            p.style = 'left: ' + minX + 'px; ' +
                'top: ' + minY + 'px; ';
             //   'width: ' + '15' + 'px; ';
            p.style = 'position: absolute'; //KOSTA*/
            const highlighter = document.createElement('div');
            highlighter.setAttribute('class', 'highlighter');
            highlighter.style = 'left: ' + minX + 'px; ' +
                'top: ' + minY + 'px; ' +
                'width: ' + width_ + 'px; ' +
                'height: ' + height_ + 'px;';
            highlighter.innerHTML = '<p>'+Math.round(score) + '% ' + 'MNM'+'</p>';

            liveView.appendChild(highlighter);
            //highlighter.appendChild(p);
            children.push(highlighter);
            //children.push(p);
            //liveView.appendChild(p);
        }
    }


}


//Load asynchronically the model of GraphModel type
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
//Loading custom pretrained cocc mobilenet_v2:
//tensorflowjs_converter --input_format=tf_saved_model --output_node_names='detection_boxes,detection_classes,detection_features,detection_multiclass_scores,detection_scores,num_detections,raw_detection_boxes,raw_detection_scores' --saved_model_tags=serve --output_format=tfjs_graph_model /content/fine_tuned_model/saved_model /content/fine_tuned_model/web_model

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




function isAbsoluteURL(str) {
    const pattern = new RegExp('^(?:[a-z]+:)?//', 'i');
    return !!pattern.test(str);
}

function getModelPath(absoluteOrRelativeUrl) {
    const modelJsonPath = isAbsoluteURL(absoluteOrRelativeUrl) ? absoluteOrRelativeUrl : window.location.pathname + absoluteOrRelativeUrl
    return modelJsonPath;
}
*/

//-----------------OLD:
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




