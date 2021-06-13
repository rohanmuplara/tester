import * as tf from "@tensorflow/tfjs";
import { drawPixelsToCanvas, runModel } from "./core";
export class End_to_End_Tops {
  models_dict: any;

  constructor() {
    this.initializeModel(this.getModelsPathDict());
  }
  getModelsPathDict(): any {
    return {
      person_detection:
        "https://storage.googleapis.com/uplara_tfjs/newest_rohan/person_detection_graph/model.json",
      denspose:
        "https://storage.googleapis.com/uplara_tfjs/newest_rohan/denspose_graph2/model.json",
      human_binary_mask:
        "https://storage.googleapis.com/uplara_tfjs/newest_rohan/human_binary_graph/model.json",
      human_parsing:
        "https://storage.googleapis.com/uplara_tfjs/newest_rohan/human_parsing_graph/model.json",
      expected_seg:
        "https://storage.googleapis.com/uplara_tfjs/newest_rohan/expected_seg_graph2/model.json",
      tps: "https://storage.googleapis.com/uplara_tfjs/newest_rohan/tps_graph/model.json",
      cloth_inpainting:
        "https://storage.googleapis.com/uplara_tfjs/newest_rohan/cloth_inpainting_graph/model.json",
      skin_inpainting:
        "https://storage.googleapis.com/uplara_tfjs/newest_rohan/skin_inpainting_graph2/model.json",
    };
  }
  async initializeModel(models_paths_dict: Map<string, string>) {
    this.models_dict = await Promise.all(
      Object.entries(models_paths_dict).map(
        async ([model_name, model_path]) => [
          model_name,
          await tf.loadGraphModel(model_path),
        ]
      )
    ).then(Object.fromEntries);
    this.complete_process();
  }

  async person_graph(person: tf.Tensor) {
    person = tf.expandDims(person, 0);
    let person_detection_output = await runModel(
      this.models_dict["person_detection"],
      { person: person },
      ["person"],
      true
    );
    let denspose_output = await runModel(
      this.models_dict["denspose"],
      { person: person_detection_output["person"] },
      ["denspose_mask"],
      true
    );
    let human_binary_mask_output = await runModel(
      this.models_dict["human_binary_mask"],
      {
        person: person_detection_output["person"],
        denspose_mask: denspose_output["denspose_mask"],
      },
      ["human_binary_mask"],
      true
    );
    let human_parsing_output = await runModel(
      this.models_dict["human_parsing"],
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
      this.models_dict["expected_seg"],
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
    let tps_output = await runModel(
      this.models_dict["tps"],
      {
        expected_seg_mask: expected_seg_output["expected_seg_mask"],
        cloth: cloth_graph_outputs["cloth"],
        cloth_mask: cloth_graph_outputs["cloth_mask"],
      },
      ["warped_cloth", "warped_cloth_mask"],
      true
    );
    let cloth_inpainting_output = await runModel(
      this.models_dict["cloth_inpainting"],
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
      this.models_dict["skin_inpainting"],
      {
        person: person_graph_outputs["person"],
        human_parsing_mask: person_graph_outputs["human_parsing_mask"],
        expected_seg_mask: expected_seg_output["expected_seg_mask"],
        inpainted_cloth: cloth_inpainting_output["inpainted_cloth"],
      },
      ["person"],
      true
    );
    drawPixelsToCanvas(skin_inpainting_output["person"]);
  }

  async complete_process() {
    debugger;
    let cloth_graph_outputs = {
      cloth_mask: tf.ones([1, 256, 192, 1]),
      cloth: tf.ones([1, 256, 192, 3]),
    };
    let person = tf.ones([256, 192, 3]);
    let person_graph_outputs = await this.person_graph(person);
    await this.tryon_graph(cloth_graph_outputs, person_graph_outputs);
  }
}
