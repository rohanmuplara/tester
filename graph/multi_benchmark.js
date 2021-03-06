async function benchmarkInput (models, num_runs) {
      cloth_graph_outputs = {"cloth_mask": tf.ones([1,256,192,1]), "cloth": tf.ones([1,256,192,3])}
      let person = tf.ones([1,256,192,3])
      let person_detection = models["person_detection"]
      let person_detection_output = await runModel(models["person_detection"], {"person": person}, ["person"], true);
      let denspose_output = await runModel(models["denspose"], {"person": person_detection_output["person"]}, ["denspose_mask"], true);
      console.log("make it pass denspose output");
      let human_binary_mask_output = await runModel(models["human_binary_mask"], {"person": person_detection_output["person"], "denspose_mask": denspose_output["denspose_mask"]}, ["human_binary_mask"], true);
      console.log("make it pass human_binary");
      let human_parsing_output = await runModel(models["human_parsing"], {"person": person_detection_output["person"], "denspose_mask": denspose_output["denspose_mask"], "human_binary_mask":  human_binary_mask_output["human_binary_mask"]}, ["person","human_parsing_mask" ], true);
      console.log("make it pass human_paring");
      let expected_seg_output = await runModel(models["expected_seg"], {"person": human_parsing_output["person"], "cloth": cloth_graph_outputs["cloth"], "cloth_mask": cloth_graph_outputs["cloth_mask"], "denspose_mask": denspose_output["denspose_mask"], "human_parsing_mask": human_parsing_output["human_parsing_mask"]},["expected_seg_mask"], true);
      console.log("make it pass expected seg");
      let tps_output = await runModel(models["tps"], {"expected_seg_mask": expected_seg_output["expected_seg_mask"], "cloth": cloth_graph_outputs["cloth"], "cloth_mask": cloth_graph_outputs["cloth_mask"]}, ["warped_cloth", "warped_cloth_mask"], true);
      console.log("make it pass tps output");
      let cloth_inpainting_output = await runModel(models["cloth_inpainting"], {"warped_cloth": tps_output["warped_cloth"], "warped_cloth_mask":tps_output["warped_cloth_mask"],"cloth": cloth_graph_outputs["cloth"], "expected_seg_mask": expected_seg_output["expected_seg_mask"]}, ["inpainted_cloth"], true);
      console.log("make it pass cloth_inpainting");
      let skin_inpainting_output = await runModel(models["skin_inpainting"], {"person": human_parsing_output["person"], "human_parsing_mask": human_parsing_output["human_parsing_mask"], "expected_seg_mask": expected_seg_output["expected_seg_mask"], "inpainted_cloth": cloth_inpainting_output["inpainted_cloth"]}, ["person"], true);
      console.log("make it pass skin inpainting");
}

function average(array) {
    let average = array.reduce((a, b) => a + b) / array.length;
    return average;
}

async function benchmarkInputDefininedInCode() {
  let model_paths_dict = {"tps": "https://storage.googleapis.com/uplara_tfjs/newest_rohan/tps_graph/model.json", 
                "person_detection": "https://storage.googleapis.com/uplara_tfjs/newest_rohan/person_detection_graph/model.json",
               "denspose":"https://storage.googleapis.com/uplara_tfjs/newest_rohan/denspose_graph2/model.json",
               "human_binary_mask": "https://storage.googleapis.com/uplara_tfjs/newest_rohan/human_binary_graph/model.json",
               "human_parsing": "https://storage.googleapis.com/uplara_tfjs/newest_rohan/human_parsing_graph/model.json",
               "expected_seg": "https://storage.googleapis.com/uplara_tfjs/newest_rohan/expected_seg_graph2/model.json",
               "tps":  "https://storage.googleapis.com/uplara_tfjs/newest_rohan/tps_graph/model.json", 
               "cloth_inpainting": "https://storage.googleapis.com/uplara_tfjs/newest_rohan/cloth_inpainting_graph/model.json",
            "skin_inpainting": "https://storage.googleapis.com/uplara_tfjs/newest_rohan/skin_inpainting_graph2/model.json"};
   let models_dict = await Promise.all(
      Object.entries(model_paths_dict).map(async ([model_name, model_path]) => [model_name, await tf.loadGraphModel(model_path)]
    )).then(Object.fromEntries);
    benchmarkInput(models_dict, 10);
}

benchmarkInputDefininedInCode();
