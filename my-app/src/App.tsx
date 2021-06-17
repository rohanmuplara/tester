import { useCallback, useEffect, useRef } from "react";
import { useDropzone } from "react-dropzone";
import "./App.css";
import { ClothandMaskPath } from "./tfjs_code/base_tfjs";
import {
  convert_file_to_img_data,
  getKeyNameFromFile,
} from "./tfjs_code/image_utils";
import { Tops_Tfjs } from "./tfjs_code/tops_tfjs";

function App() {
  const refContainer = useRef<Tops_Tfjs>();
  useEffect(() => {
    // Update the document title using the browser API
    refContainer.current = new Tops_Tfjs();
    return refContainer.current.disposeModelFromGpu();
  }, []);

  const onDrop = useCallback(async (uploadedFiles) => {
    console.log("the accepted files are" + uploadedFiles);

    let person_image_data_url = await convert_file_to_img_data(
      uploadedFiles[0]
    ).then(
      (data) => data,
      (error) => console.log(error + "we don't support this file type")
    );
    let person_key = getKeyNameFromFile(uploadedFiles[0]);
    let cloths_path_array = [
      [
        "https://storage.googleapis.com/uplara_tfjs/cloth_images/c/cloth.png",
        "https://storage.googleapis.com/uplara_tfjs/cloth_images/c/cloth_mask.png",
      ],
    ] as ClothandMaskPath[];
    refContainer.current!.runTryon(
      cloths_path_array,
      person_key,
      person_image_data_url!
    );
    refContainer.current!.runTryon(cloths_path_array, person_key);
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
