import * as tf from "@tensorflow/tfjs-core";
import * as tfc from "@tensorflow/tfjs-converter";
import "@tensorflow/tfjs-backend-webgl";

import { NamedTensorMap } from "@tensorflow/tfjs-core";
import {
  convertMaskUrlToTensor,
  convertImageUrlToTensor,
  converTensorToDataUrls,
  convertDataUrlsToTensor,
} from "./core";
import { Tensor_Storage_Map } from "./tensor_storage_map";

// copied naming patterns from tfjs
export type NamedModelMap = {
  [name: string]: tfc.GraphModel;
};

export type NamedModelPathMap = {
  [name: string]: string;
};

export type ClothandMaskPath = [string, string];
export abstract class BaseTfjs {
  models_map: Map<string, tfc.GraphModel> | undefined;

  models_present_indexdb_set: Set<string>;

  models_ready: boolean;

  person_graph_output_map: Tensor_Storage_Map;

  abstract getModelsPathDict(): NamedModelPathMap;

  // dummy  abstract because can't declare abstract async methods
  async tryon_graph(
    cloth_graph_outputs: NamedTensorMap,
    person_graph_outputs: NamedTensorMap
  ): Promise<NamedTensorMap> {
    return { "not implemented": tf.tensor(0.0) };
  }
  // dummy  abstract because can't declare abstract async methods
  async person_graph(person: NamedTensorMap): Promise<NamedTensorMap> {
    return { "not implemented": tf.tensor(0.0) };
  }

  // should never have to change this but just in case we change inputs cloth_graph gives
  get_cloth_graph_dummy_outputs(): NamedTensorMap {
    return {
      cloth_mask: tf.ones([1, 256, 192, 1]),
      cloth: tf.ones([1, 256, 192, 3]),
    };
  }

  // should never have to change this but just in case input person ie shape
  get_person_input_dummy(): NamedTensorMap {
    return { person: tf.ones([1, 256, 192, 3]) };
  }

  constructor() {
    this.models_present_indexdb_set = new Set<string>();
    this.models_ready = false;

    this.person_graph_output_map = new Tensor_Storage_Map(
      "person_graph_output_map",
      3
    );

    this.initializeProcess();
  }

  async initializeProcess() {
    let modelsPath = this.getModelsPathDict();
    await this.initializeModels(modelsPath);
    await this.runModelWithDummyInputs();
    this.download_models_to_index_db();
  }

  /**
   *
   * have seperate dic for in disk so caller doesn't have to worry about this
   */
  async initializeModels(models_paths_dict: Object) {
    let models_entries = (await Promise.all(
      Object.entries(models_paths_dict).map(
        async ([model_name, model_path]) => {
          let index_path = "indexeddb://" + model_name;
          let model = await tfc.loadGraphModel(index_path).then(
            (value: tfc.GraphModel) => {
              this.models_present_indexdb_set.add(model_name);
              return value;
            },
            (_) => {
              return tfc.loadGraphModel(model_path);
            }
          );
          return [model_name, model];
        }
      )
    )) as any;
    this.models_map = new Map(models_entries);
    this.models_ready = true;
  }
  download_models_to_index_db() {
    this.models_map!.forEach((model, model_name) => {
      if (!this.models_present_indexdb_set.has(model_name)) {
        console.log("the model name is" + model_name);
        let index_path = "indexeddb://" + model_name;
        model.save(index_path);
      }
    });
  }

  /**
   */
  /**
   * the first time running something is super slow because model has to be initialized on gpu.
   * so we warm up the model by running model once with dummy inputs.
   */

  async runModelWithDummyInputs() {
    console.log("we are in run model with dummy inputs");
    let cloth_graph_output = this.get_cloth_graph_dummy_outputs();
    let person_input = this.get_person_input_dummy();
    let person_graph_output = await this.person_graph(person_input);
    tf.dispose(person_input);
    let tryon_graph_output = await this.tryon_graph(
      cloth_graph_output,
      person_graph_output
    );
    tf.dispose(cloth_graph_output);
    tf.dispose(person_graph_output);
    tf.dispose(tryon_graph_output);
    console.log("we are done running dumm inputs");
  }

  async runTryon(
    clothsAndMasksPath: ClothandMaskPath[],
    person_key: string,
    person_data_url?: string
  ): Promise<string[]> {
    let cloth_path = clothsAndMasksPath[0][0];
    let cloth_mask_path = clothsAndMasksPath[0][1];

    let person_graph_output =
      await this.person_graph_output_map.getNamedTensorMap(person_key);
    await this.ensureChecks();

    if (person_graph_output === null) {
      if (person_data_url) {
        let person_tensor = await convertDataUrlsToTensor([person_data_url]);
        let person_inputs = {
          person: person_tensor as tf.Tensor4D,
        };
        person_graph_output = await this.person_graph(person_inputs);
        await this.person_graph_output_map.setNameTensorMap(
          person_key,
          person_graph_output
        );
      } else {
        return Promise.reject("Person key does not exist");
      }
    }
    let cloths_tensor = await convertImageUrlToTensor([cloth_path]);
    let cloths_mask_tensor = await convertMaskUrlToTensor([cloth_mask_path]);
    let cloth_graph_output: NamedTensorMap = {
      cloth_mask: cloths_mask_tensor,
      cloth: cloths_tensor,
    };

    let tryon_graph_output = await this.tryon_graph(
      cloth_graph_output,
      person_graph_output
    );

    tf.dispose(cloth_graph_output);
    tf.dispose(person_graph_output);

    let tryon_person_data_array = await converTensorToDataUrls(
      tryon_graph_output["person"] as tf.Tensor4D
    );
    tf.dispose(tryon_graph_output);
    return tryon_person_data_array;
  }

  disposeModelFromGpu(): void {
    if (this.models_map) {
      this.models_map.forEach((model: tfc.GraphModel, _) => {
        model.dispose();
      });
    }
  }

  async ensureChecks(): Promise<void> {
    while (!this.models_ready) {
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    return;
  }

  getPersonKeys(): string[] {
    return this.person_graph_output_map.getExistingKeys();
  }
}
