import { useCallback, useEffect, useRef } from "react";
import { useDropzone } from "react-dropzone";
import "./App.css";
import { convert_files_to_img_data } from "./tfjs_code/image_utils";
import { Tops_Tfjs } from "./tfjs_code/tops_tfjs";

function App() {
  const refContainer = useRef<Tops_Tfjs>();
  useEffect(() => {
    // Update the document title using the browser API
    refContainer.current = new Tops_Tfjs();
    return refContainer.current.disposeModelFromGpu();
  }, []);

  const onDrop = useCallback(async (acceptedFiles) => {
    console.log("the accepted files are" + acceptedFiles);
    let cloths_path = [
      "https://storage.googleapis.com/uplara_tfjs/cloth_images/c/cloth.png",
    ];
    let cloth_masks_path = [
      "https://storage.googleapis.com/uplara_tfjs/cloth_images/c/cloth_mask.png",
    ];

    let person_image_data_urls = await convert_files_to_img_data(acceptedFiles);
    refContainer.current!.runModelWithNewPerson(
      cloths_path,
      cloth_masks_path,
      ["dummy_person"],
      person_image_data_urls[0]
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
