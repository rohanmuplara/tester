async function benchmarkInput (model_path, tensors, num_runs) {

  let model_loading_begin = window.performance.now();
  let model = await tf.loadGraphModel(model_path);
  let model_loading_end = window.performance.now();
  let model_loading_time = model_loading_end - model_loading_begin
  console.log2("model loading time", model_loading_end - model_loading_begin);
  let first_prediction_begin = window.performance.now();
  const predictions = await runModel(model, tensors, false);
  let first_prediction_end = window.performance.now();
  let first_prediction_time = first_prediction_end - first_prediction_begin;
  console.log2("first prediction time" + first_prediction_time);

  let subsequent_times =new Float32Array(num_runs - 1);
  for (let i = 0; i < num_runs - 1 ; i++) {
    let begin= window.performance.now();
    const predictions = await runModel(model, tensors, false);
    let end= window.performance.now();
    let time = (end-begin) ;
    subsequent_times[i] = time;
  }
  console.log("the subsequent times are", subsequent_times);
  console.log2("subsequent predictions are in ms" + subsequent_times);
  let average_time = average(subsequent_times);

  console.log2("the average is" +  average_time);
}

function average(array) {
    let average = array.reduce((a, b) => a + b) / array.length;
    return average;
}

function benchmarkInputDefininedInCode() {
    let tensor1 = tf.ones([1,224, 224, 3]);
    benchmarkInput("https://storage.googleapis.com/uplara_tfjs/seperatemobilenet/model.json", [tensor1], 100);
}

//benchmarkInputDefininedInCode();

(function () {
    if (!console) {
        console = {};
    }
    var old = console.log2;
    var logger = document.getElementById('log');
    console.log2 = function (message) {
        if (typeof message == 'object') {
            logger.innerHTML += (JSON && JSON.stringify ? JSON.stringify(message) : String(message)) + '<br />';
        } else {
            logger.innerHTML += message + '<br />';
        }
    }
})();
