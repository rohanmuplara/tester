import * as tf from "@tensorflow/tfjs";
import { drawPixelsToCanvas, runModel } from "./core";
class End_to_End_Tops {
  models_paths_dict: [string: string];
  constructor(models_paths_dict: [string: string]) {
    this.models_paths_dict = models_paths_dict;
  }
  async process() {
    let models = await Promise.all(
      Object.entries(this.models_paths_dict).map(
        async ([model_name, model_path]) => [
          model_name,
          await tf.loadGraphModel(model_path),
        ]
      )
    ).then(Object.fromEntries);
    let cloth_graph_outputs = {
      cloth_mask: tf.ones([1, 256, 192, 1]),
      cloth: tf.ones([1, 256, 192, 3]),
    };
    let person = tf.ones([1, 256, 192, 3]);
    let person_detection_output = await runModel(
      models["person_detection"],
      { person: person },
      ["person"],
      true
    );
    let denspose_output = await runModel(
      models["denspose"],
      { person: person_detection_output["person"] },
      ["denspose_mask"],
      true
    );
    let human_binary_mask_output = await runModel(
      models["human_binary_mask"],
      {
        person: person_detection_output["person"],
        denspose_mask: denspose_output["denspose_mask"],
      },
      ["human_binary_mask"],
      true
    );
    let human_parsing_output = await runModel(
      models["human_parsing"],
      {
        person: person_detection_output["person"],
        denspose_mask: denspose_output["denspose_mask"],
        human_binary_mask: human_binary_mask_output["human_binary_mask"],
      },
      ["person", "human_parsing_mask"],
      true
    );
    let expected_seg_output = await runModel(
      models["expected_seg"],
      {
        person: human_parsing_output["person"],
        cloth: cloth_graph_outputs["cloth"],
        cloth_mask: cloth_graph_outputs["cloth_mask"],
        denspose_mask: denspose_output["denspose_mask"],
        human_parsing_mask: human_parsing_output["human_parsing_mask"],
      },
      ["expected_seg_mask"],
      true
    );
    let tps_output = await runModel(
      models["tps"],
      {
        expected_seg_mask: expected_seg_output["expected_seg_mask"],
        cloth: cloth_graph_outputs["cloth"],
        cloth_mask: cloth_graph_outputs["cloth_mask"],
      },
      ["warped_cloth", "warped_cloth_mask"],
      true
    );
    let cloth_inpainting_output = await runModel(
      models["cloth_inpainting"],
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
      models["skin_inpainting"],
      {
        person: human_parsing_output["person"],
        human_parsing_mask: human_parsing_output["human_parsing_mask"],
        expected_seg_mask: expected_seg_output["expected_seg_mask"],
        inpainted_cloth: cloth_inpainting_output["inpainted_cloth"],
      },
      ["person"],
      true
    );
    drawPixelsToCanvas(skin_inpainting_output["person"]);
  }
}
