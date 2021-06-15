import { useCallback, useEffect, useRef } from "react";
import { useDropzone } from "react-dropzone";
import "./App.css";
import { convert_file_to_img } from "./tfjs_code/image_utils";
import { Tops_Tfjs } from "./tfjs_code/tops_tfjs";
import * as tf from "@tensorflow/tfjs";

function App() {
  const refContainer = useRef<Tops_Tfjs>();
  useEffect(() => {
    // Update the document title using the browser API
    refContainer.current = new Tops_Tfjs();
    return refContainer.current.disposeModelFromGpu();
  }, []);

  const onDrop = useCallback(async (acceptedFiles) => {
    console.log("the accepted files are" + acceptedFiles);
    let cloth_path =
      "https://storage.googleapis.com/uplara_tfjs/cloth_images/a/cloth_raw.png";
    let cloth_mask_path =
      "https://storage.googleapis.com/uplara_tfjs/cloth_images/a/cloth_mask_raw.png";

    let person_image = await convert_file_to_img(acceptedFiles[0]);
    let person_tensor = tf.browser.fromPixels(person_image, 3);
    refContainer.current!.runModel(
      cloth_path,
      cloth_mask_path,
      "dummy_person",
      person_tensor
    );
  }, []);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });
  return (
    <div {...getRootProps()}>
      <input {...getInputProps()} />
      {isDragActive ? (
        <p>Drop the files here ...</p>
      ) : (
        <p>Drag 'n' drop some files here, or click to select files</p>
      )}
    </div>
  );
}

export default App;
