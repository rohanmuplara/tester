async function benchmarkInput (model_path, tensors, num_runs) {

  let model_loading_begin = window.performance.now();
  let model = await tf.loadGraphModel(model_path);
  let model_loading_end = window.performance.now();
  console.log(model_loading_end - model_loading_begin);
  let first_prediction_begin = window.performance.now();
  const predictions = await runModel(model, tensors, false);
  let first_prediction_end = window.performance.now();
  console.log(first_prediction_end - first_prediction_begin);

  let subsequent_times =new Float32Array(num_runs - 1);
  for (let i = 0; i < num_runs - 1 ; i++) {
    let begin= window.performance.now();
    const predictions = await runModel(model, tensors, false);
    let end= window.performance.now();
    let time = (end-begin) ;
    subsequent_times[i] = time;
  }
  console.log("subsequent predictions are in ms", subsequent_times);
  console.log("the average of the subequent predictions are", average(subsequent_times));
}

function average(array) {
    let average = array.reduce((a, b) => a + b) / array.length;
    return average;
}

function benchmarkInputDefininedInCode() {
    let tensor1 = tf.ones([1,224, 224, 3]);
    benchmarkInput("https://storage.googleapis.com/uplara_tfjs/seperatemobilenet/model.json", [tensor1], 100);
}

benchmarkInputDefininedInCode();

(function () {
    if (!console) {
        console = {};
    }
    var old = console.log;
    var logger = document.getElementById('log');
    console.log = function (message) {
        if (typeof message == 'object') {
            logger.innerHTML += (JSON && JSON.stringify ? JSON.stringify(message) : String(message)) + '<br />';
        } else {
            logger.innerHTML += message + '<br />';
        }
    }
})();
