import * as tf from "@tensorflow/tfjs";
import { models } from "@tensorflow/tfjs";
import {
  drawPixelsToCanvas,
  handle_image_load,
  runModel,
  convertMaskUrlToTensor,
  convertImageUrlToTensor,
  convertMaskToColors,
} from "./core";
export class End_to_End_Tops {
  models_map: Map<string, tf.GraphModel>;

  constructor() {
    this.initializeModel(this.getModelsPathDict());
    this.models_map = new Map<string, tf.GraphModel>();
  }
  getModelsPathDict(): Map<string, string> {
    let model_names_map = new Map<string, string>([
      [
        "person_detection",
        "https://storage.googleapis.com/uplara_tfjs/newest_rohan/person_detection_graph/model.json",
      ],
      [
        "denspose",
        "https://storage.googleapis.com/uplara_tfjs/newest_rohan/denspose_graph2/model.json",
      ],
      [
        "human_binary_mask",
        "https://storage.googleapis.com/uplara_tfjs/newest_rohan/human_binary_graph/model.json",
      ],
      [
        "human_parsing",
        "https://storage.googleapis.com/uplara_tfjs/newest_rohan/human_parsing_graph/model.json",
      ],
      [
        "expected_seg",
        "https://storage.googleapis.com/uplara_tfjs/newest_rohan/expected_seg_graph2/model.json",
      ],
      [
        "tps",
        "https://storage.googleapis.com/uplara_tfjs/newest_rohan/tps_graph/model.json",
      ],
      [
        "cloth_inpainting",
        "https://storage.googleapis.com/uplara_tfjs/newest_rohan/cloth_inpainting_graph/model.json",
      ],
      [
        "skin_inpainting",
        "https://storage.googleapis.com/uplara_tfjs/newest_rohan/skin_inpainting_graph2/model.json",
      ],
    ]);
    return model_names_map;
  }
  async initializeModel(models_paths_dict: Map<string, string>) {
    
   models_paths_dict.entries().map(model_name: string, model_path:string) => {
        let model = await tf.loadGraphModel(model_path)
        return model_name, model_path
   })
   
  }
  async person_graph(person: tf.Tensor) {
    person = tf.expandDims(person, 0);
    person = tf.cast(person, "float32");
    let person_detection_output = await runModel(
      this.models_map.get("person_detection")!,
      { person: person },
      ["person"],
      true
    );
    let denspose_output = await runModel(
      this.models_map.get("denspose")!,
      { person: person_detection_output["person"] },
      ["denspose_mask"],
      true
    );
    let human_binary_mask_output = await runModel(
      this.models_map.get("human_binary_mask")!,
      {
        person: person_detection_output["person"],
        denspose_mask: denspose_output["denspose_mask"],
      },
      ["human_binary_mask"],
      true
    );
    let human_parsing_output = await runModel(
      this.models_map.get("human_parsing")!,
      {
        person: person_detection_output["person"],
        denspose_mask: denspose_output["denspose_mask"],
        human_binary_mask: human_binary_mask_output["human_binary_mask"],
      },
      ["person", "human_parsing_mask"],
      true
    );
    return Object.assign({}, human_parsing_output, denspose_output);
  }

  async tryon_graph(cloth_graph_outputs: any, person_graph_outputs: any) {
    let expected_seg_output = await runModel(
      this.models_map.get("expected_seg")!,
      {
        person: person_graph_outputs["person"],
        cloth: cloth_graph_outputs["cloth"],
        cloth_mask: cloth_graph_outputs["cloth_mask"],
        denspose_mask: person_graph_outputs["denspose_mask"],
        human_parsing_mask: person_graph_outputs["human_parsing_mask"],
      },
      ["expected_seg_mask"],
      true
    );
    drawPixelsToCanvas(
      convertMaskToColors(expected_seg_output["expected_seg_mask"]),
      "expected_seg_output.png"
    );
    let tps_output = await runModel(
      this.models_map.get("tps")!,
      {
        expected_seg_mask: expected_seg_output["expected_seg_mask"],
        cloth: cloth_graph_outputs["cloth"],
        cloth_mask: cloth_graph_outputs["cloth_mask"],
      },
      ["warped_cloth", "warped_cloth_mask"],
      true
    );
    drawPixelsToCanvas(tps_output["warped_cloth"], "warped_cloth.png");
    let cloth_inpainting_output = await runModel(
      this.models_map.get("cloth_inpainting")!,
      {
        warped_cloth: tps_output["warped_cloth"],
        warped_cloth_mask: tps_output["warped_cloth_mask"],
        cloth: cloth_graph_outputs["cloth"],
        expected_seg_mask: expected_seg_output["expected_seg_mask"],
      },
      ["inpainted_cloth"],
      true
    );
    let skin_inpainting_output = await runModel(
      this.models_map.get("skin_inpainting")!,
      {
        person: person_graph_outputs["person"],
        human_parsing_mask: person_graph_outputs["human_parsing_mask"],
        expected_seg_mask: expected_seg_output["expected_seg_mask"],
        inpainted_cloth: cloth_inpainting_output["inpainted_cloth"],
      },
      ["person"],
      true
    );
    drawPixelsToCanvas(
      skin_inpainting_output["person"],
      "skin_inpainting_output.png"
    );
  }

  async complete_process() {
    let cloth_graph_outputs = {
      cloth_mask: tf.ones([1, 256, 192, 1]),
      cloth: tf.ones([1, 256, 192, 3]),
    };
    let person = tf.ones([256, 192, 3]);
    let person_graph_outputs = await this.person_graph(person);
    await this.tryon_graph(cloth_graph_outputs, person_graph_outputs);
  }

  async handle_person_upload(files: [any]) {
    let person_tensor = await handle_image_load(files);
    let cloth_tensor = await convertImageUrlToTensor(
      "https://storage.googleapis.com/uplara_tfjs/cloth_images/a/cloth_raw.png"
    );
    cloth_tensor = tf.expandDims(cloth_tensor, 0);
    cloth_tensor = tf.cast(cloth_tensor, "float32");
    let cloth_mask_tensor = await convertMaskUrlToTensor(
      "https://storage.googleapis.com/uplara_tfjs/cloth_images/a/cloth_mask_raw.png"
    );
    cloth_mask_tensor = tf.expandDims(cloth_mask_tensor);
    cloth_mask_tensor = tf.div(cloth_mask_tensor, 51);
    cloth_mask_tensor = tf.cast(cloth_mask_tensor, "float32");
    let cloth_graph_outputs = {
      cloth_mask: cloth_mask_tensor,
      cloth: cloth_tensor,
    };
    cloth_mask_tensor = tf.expandDims(cloth_mask_tensor, 0);
    let person_graph_outputs = await this.person_graph(person_tensor);
    let tryon_outputs = this.tryon_graph(
      cloth_graph_outputs,
      person_graph_outputs
    );
  }
}
