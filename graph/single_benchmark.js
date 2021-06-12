async function benchmarkInput (model_path, tensors, num_runs) {
  console.log("loading the cloth seg model")
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
    tps_inputs = {"cloth": tf.ones([1,256, 192, 3]), "human_parsing_mask": tf.ones([1,256, 192, 1]), "denspose_mask": tf.ones([1,256, 192, 3]),
      "cloth_mask": tf.ones([1,256, 192, 1]),  "expected_seg_mask": tf.ones([1,256, 192, 1])}
    sample_inputs = {"a": tf.ones([5, 1])}
    //benchmarkInput("https://storage.googleapis.com/uplara_tfjs/newest_rohan/tps_graph/model.json", tps_inputs, 15)
   person_detection_inputs = {"person": tf.ones([1,256,193,3])}
   // benchmarkInput("https://storage.googleapis.com/uplara_tfjs/newest_rohan/person_detection_graph/model.json", person_detection_inputs, 15)
   //denspose_inputs = {"person": tf.ones([1,256,192,3])}
   //benchmarkInput("https://storage.googleapis.com/uplara_tfjs/newest_rohan/denspose_graph/model.json", denspose_inputs, 15)
   human_binary_mask_inputs = {"person": tf.ones([1,256, 192, 3]), "denspose_mask": tf.ones([1,256, 192, 1])}
   //benchmarkInput("https://storage.googleapis.com/uplara_tfjs/newest_rohan/human_binary_graph/model.json", human_binary_mask_inputs, 15)
   human_parsing_inputs =  {"person": tf.ones([1,256, 192, 3]), "human_binary_mask": tf.ones([1,256, 192, 1]), "denspose_mask": tf.ones([1,256, 192, 1])}
   //benchmarkInput("https://storage.googleapis.com/uplara_tfjs/newest_rohan/human_parsing_graph/model.json", human_parsing_inputs, 15)
   expected_seg_inputs =  {"person": tf.ones([1,256, 192, 3]), "human_parsing_mask": tf.ones([1,256, 192, 1]), "denspose_mask": tf.ones([1,256, 192, 1]), "cloth": tf.ones([1,256, 192, 3]), "cloth_mask": tf.ones([1,256, 192, 1])}
   //benchmarkInput("https://storage.googleapis.com/uplara_tfjs/newest_rohan/expected_seg_graph/model.json", expected_seg_inputs, 15)
   cloth_inpainting_inputs =  {"cloth": tf.ones([1,256, 192, 3]), "warped_cloth": tf.ones([1,256, 192, 3]),  "warped_cloth_mask": tf.ones([1,256, 192, 1]), 
   "expected_seg_mask": tf.ones([1,256, 192, 1])}
   benchmarkInput("https://storage.googleapis.com/uplara_tfjs/newest_rohan/cloth_inpainting_graph/model.json", cloth_inpainting_inputs, 15)
}

benchmarkInputDefininedInCode();

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

