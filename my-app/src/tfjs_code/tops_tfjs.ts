import * as tf from "@tensorflow/tfjs-core";
import { NamedTensorMap } from "@tensorflow/tfjs-core";
import { BaseTfjs, NamedModelPathMap } from "./base_tfjs";
import { runModel } from "./core";
export class Tops_Tfjs extends BaseTfjs {
  getModelsPathDict(): NamedModelPathMap {
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

  async person_graph(
    person_graph_inputs: NamedTensorMap
  ): Promise<NamedTensorMap> {
    let person = tf.cast(person_graph_inputs["person"] as tf.Tensor, "float32");
    let person_detection_output = await runModel(
      this.models_map!.get("person_detection")!,
      { person: person },
      ["person"],
      false
    );
    let denspose_output = await runModel(
      this.models_map!.get("denspose")!,
      { person: person_detection_output["person"] },
      ["denspose_mask"],
      true
    );
    let human_binary_mask_output = await runModel(
      this.models_map!.get("human_binary_mask")!,
      {
        person: person_detection_output["person"],
        denspose_mask: denspose_output["denspose_mask"],
      },
      ["human_binary_mask"],
      true
    );
    let human_parsing_output = await runModel(
      this.models_map!.get("human_parsing")!,
      {
        person: person_detection_output["person"],
        denspose_mask: denspose_output["denspose_mask"],
        human_binary_mask: human_binary_mask_output["human_binary_mask"],
      },
      ["person", "human_parsing_mask"],
      true
    );
    tf.dispose(person_detection_output);
    tf.dispose(human_binary_mask_output);
    return Object.assign({}, human_parsing_output, denspose_output);
  }

  async tryon_graph(
    cloth_graph_outputs: NamedTensorMap,
    person_graph_outputs: NamedTensorMap
  ): Promise<NamedTensorMap> {
    let expected_seg_output = await runModel(
      this.models_map!.get("expected_seg")!,
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
      this.models_map!.get("tps")!,
      {
        expected_seg_mask: expected_seg_output["expected_seg_mask"],
        cloth: cloth_graph_outputs["cloth"],
        cloth_mask: cloth_graph_outputs["cloth_mask"],
      },
      ["warped_cloth", "warped_cloth_mask"],
      false
    );
    let cloth_inpainting_output = await runModel(
      this.models_map!.get("cloth_inpainting")!,
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
      this.models_map!.get("skin_inpainting")!,
      {
        person: person_graph_outputs["person"],
        human_parsing_mask: person_graph_outputs["human_parsing_mask"],
        expected_seg_mask: expected_seg_output["expected_seg_mask"],
        inpainted_cloth: cloth_inpainting_output["inpainted_cloth"],
      },
      ["person"],
      true
    );
    tf.dispose(expected_seg_output);
    tf.dispose(tps_output);
    tf.dispose(cloth_inpainting_output);
    return skin_inpainting_output;
  }
}
