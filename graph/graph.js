async function testTensor (tensor, num_runs) {
  console.time("model loading time");
  const model = await tf.loadGraphModel("https://storage.googleapis.com/uplara_tfjs/final5/model.json");
  console.timeEnd("model loading time");
  console.time("first prediction");
  const predictionsTensor =  await model.executeAsync([tensor]);
  const predictions = predictionsTensor.dataSync();
  console.timeEnd("first prediction");
  let subsequent_times =new Float32Array(num_runs - 1);
  for (let i = 0; i < num_runs - 1 ; i++) {
    var begin= window.performance.now();
    const predictionsTensor =  await model.executeAsync([tensor]);
    const predictions = predictionsTensor.dataSync();
    var end= window.performance.now();
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

function testTensorDefininedInCode() {
    let tensor = tf.ones([28, 28]);
    tensor = tensor.expandDims(0);
    console.log("the tensor is", tensor);
    testTensor(tensor, 10);
}

function updateValue(e) {
  console.log(e.target.value);
}

function initializeFileUploader() {
  const input = document.querySelector('input');
 input.addEventListener('input', getFile);

}

function getFile(event) {
	const input = event.target
  if ('files' in input && input.files.length > 0) {
	  placeFileContent(
      input.files[0])
  }
}

function placeFileContent(file) {
	readFileContent(file).then(content => {
    debugger;
  }).catch(error => console.log(error))
}

function readFileContent(file) {
	const reader = new FileReader()
  return new Promise((resolve, reject) => {
    reader.onload = event => resolve(event.target.result)
    reader.onerror = error => reject(error)
    reader.readAsText(file)
  })
}


initializeFileUploader();
testTensorDefininedInCode();
