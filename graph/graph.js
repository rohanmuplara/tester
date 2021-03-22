async function testTensor () {
  console.time("a");
  const tensor1 = tf.ones([256, 28, 28]);
  const new_tensor1 = tensor1.expandDims(0);
  const model = await tf.loadGraphModel("https://storage.googleapis.com/uplara_tfjs/final5/model.json");
  console.timeEnd("a");
  const predictions =  model.execute([tensor1]);
  console.log("predictions");
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
testTensor();
