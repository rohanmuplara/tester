async function testInput (tensors, num_runs) {
  console.time("model loading time");
  const model = await tf.loadGraphModel("https://storage.googleapis.com/uplara_tfjs/multipleoutputs/model.json");
  console.timeEnd("model loading time");
  console.time("first prediction");
  const predictionsTensor =  await model.executeAsync(tensors);
  debugger;
  const predictions = predictionsTensor.ArraySync();
  console.timeEnd("first prediction");
  console.log("the predictions are", predictions);
  download("results.json", predictions);
  let subsequent_times =new Float32Array(num_runs - 1);
  for (let i = 0; i < num_runs - 1 ; i++) {
    let begin= window.performance.now();
    const predictionsTensor =  await model.executeAsync(tensors);
    const predictions = predictionsTensor.ArraySync();
    let end= window.performance.now();
    let time = (end-begin) ;
    subsequent_times[i] = time;
  }
  console.log("subsequent predictions are in ms", subsequent_times);
  console.log("the average of the subequent predictions are", average(subsequent_times));
}

function download(filename, text) {
    var element = document.createElement('a');

    element.setAttribute('href', 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(text)));
    element.setAttribute('download', filename);

    element.style.display = 'none';
    document.body.appendChild(element);

    element.click();

    document.body.removeChild(element);
}

function average(array) {
    let average = array.reduce((a, b) => a + b) / array.length;
    return average;
}

function testInputDefininedInCode() {
    let tensor1 = tf.ones([5]);
    tensor1 = tensor1.expandDims(0);
    let tensor2 = tf.ones([5]);
    tensor2 = tensor2.expandDims(0);
    testInput([tensor1, tensor2], 10);
}

function testUploadedTensor(array) {
    let tensor = tf.tensor(array);
    tensor = tensor.expandDims(0);
    console.log("the tensor insider is test uploaded array is", tensor);
    testInput([tensor], 10);
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
    let values = JSON.parse(content);
    testUploadedTensor(values);
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
testInputDefininedInCode();
