import * as tf from "@tensorflow/tfjs";
import { NamedTensorMap } from "@tensorflow/tfjs";
import {
  convertMaskUrlToTensor,
  convertImageUrlToTensor,
  converTensorToDataUrls,
} from "./core";
import { Storage_Map } from "./storage_map";

// copied naming patterns from tfjs
export type NamedModelMap = {
  [name: string]: tf.GraphModel;
};

export type NamedModelPathMap = {
  [name: string]: string;
};
export abstract class BaseTfjs {
  models_map: Map<string, tf.GraphModel> | undefined;

  models_present_indexdb_set: Set<string>;

  models_ready: boolean;

  cloths_path_map: Storage_Map;

  cloths_mask_path_map: Storage_Map;

  persons_path_map: Storage_Map;

  tryons_path_map: Storage_Map;

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
    this.cloths_path_map = new Storage_Map("cloths");
    this.cloths_mask_path_map = new Storage_Map("cloths_mask");
    this.persons_path_map = new Storage_Map("person");
    this.tryons_path_map = new Storage_Map("tryons");
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
          let model = await tf.loadGraphModel(index_path).then(
            (value: tf.GraphModel) => {
              //this.models_present_indexdb_set.add(model_name);
              return value;
            },
            (_) => {
              return tf.loadGraphModel(model_path);
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
    let cloth_graph_outputs = this.get_cloth_graph_dummy_outputs();
    let person_input = this.get_person_input_dummy();
    let person_graph_outputs = await this.person_graph(person_input);
    tf.dispose(person_input);
    let tryon_graphs = await this.tryon_graph(
      cloth_graph_outputs,
      person_graph_outputs
    );
    tf.dispose(cloth_graph_outputs);
    tf.dispose(person_graph_outputs);
    tf.dispose(tryon_graphs);
  }

  async runModelWithNewPerson(
    cloths_path: string[],
    cloths_mask_path: string[],
    person_key: string,
    person_data_url?: string
  ): Promise<string[]> {
    {
      let cloth_path = cloths_path[0];
      let cloth_mask_path = cloths_mask_path[0];
      let tryon_path = cloth_path + ":" + person_key;
      let tryon_image_data = this.tryons_path_map.getItem(tryon_path);
      if (tryon_image_data !== null) {
        return tryon_image_data;
      } else {
        let cloths_mask_array =
          this.cloths_mask_path_map.getItem(cloth_mask_path);
        let cloths_mask_tensor;
        if (cloths_mask_array === null) {
          cloths_mask_tensor = await convertMaskUrlToTensor([cloth_mask_path]);
        } else {
          cloths_mask_tensor = tf.tensor(cloths_mask_array);
        }
        let cloths_array = this.cloths_path_map.getItem(cloth_path);
        let cloths_tensor;
        if (cloths_array === null) {
          cloths_tensor = await convertMaskUrlToTensor([cloth_path]);
        } else {
          cloths_tensor = tf.tensor(cloths_array);
        }
        let cloth_graph_outputs = {
          cloth_mask: cloths_mask_tensor,
          cloth: cloths_tensor,
        };
        let person_array = this.persons_path_map.getItem(person_key);
        let person_tensor;
        if (person_array === null) {
          cloths_tensor = await convertMaskUrlToTensor([cloth_path]);
        } else {
          person_tensor = tf.tensor(person_array);
        }
        let person_inputs = {
          person: person_tensor as tf.Tensor4D,
        };
        await this.ensureChecks();
        let person_graph_outputs = await this.person_graph(person_inputs);
        let tryon_graph_outputs = await this.tryon_graph(
          cloth_graph_outputs,
          person_graph_outputs
        );
        tf.dispose(cloth_graph_outputs);
        tf.dispose(person_graph_outputs);
        let tryon_person_data_array = await converTensorToDataUrls(
          tryon_graph_outputs["person"] as tf.Tensor4D
        );
        tf.dispose(tryon_graph_outputs);
        return tryon_person_data_array;
      }
    }
  }

  disposeModelFromGpu(): void {
    if (this.models_map) {
      this.models_map.forEach((model: tf.GraphModel, _) => {
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

  getPersonImages(): string[] {
    return this.persons_path_map.getExistingKeys();
  }
}
