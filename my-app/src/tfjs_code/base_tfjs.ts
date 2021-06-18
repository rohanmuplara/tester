import * as tf from "@tensorflow/tfjs";
import { NamedTensorMap } from "@tensorflow/tfjs";
import {
  convertMaskUrlToTensor,
  convertImageUrlToTensor,
  converTensorToDataUrls,
  convertDataUrlsToTensor,
} from "./core";
import { Tensor_Storage_Map } from "./tensor_storage_map";

// copied naming patterns from tfjs
export type NamedModelMap = {
  [name: string]: tf.GraphModel;
};

export type NamedModelPathMap = {
  [name: string]: string;
};

export type ClothandMaskPath = [string, string];
export abstract class BaseTfjs {
  models_map: Map<string, tf.GraphModel> | undefined;

  models_present_indexdb_set: Set<string>;

  models_ready: boolean;

  cloth_graph_output_map: Tensor_Storage_Map;

  person_graph_output_map: Tensor_Storage_Map;

  tryon_graph_output_map: Tensor_Storage_Map;

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
    this.cloth_graph_output_map = new Tensor_Storage_Map(
      "cloth_graph_output_map"
    );
    this.person_graph_output_map = new Tensor_Storage_Map(
      "person_graph_output_map"
    );
    this.tryon_graph_output_map = new Tensor_Storage_Map(
      "tryons_graph_output_map"
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
  }

  async runTryon(
    clothsAndMasksPath: ClothandMaskPath[],
    person_key: string,
    person_data_url?: string
  ): Promise<string[]> {
    let cloth_path = clothsAndMasksPath[0][0];
    let cloth_mask_path = clothsAndMasksPath[0][1];
    let cloth_key = cloth_path + ":" + cloth_mask_path;
    let tryon_key = cloth_path + ":" + person_key;
    let tryon_graph_output =
      await this.tryon_graph_output_map.getNamedTensorMap(tryon_key);
    if (tryon_graph_output !== null) {
      return await converTensorToDataUrls(
        tryon_graph_output["person"] as tf.Tensor4D
      );
    } else {
      let person_graph_output =
        await this.person_graph_output_map.getNamedTensorMap(person_key);
      await this.ensureChecks();

      if (person_graph_output === null) {
        let person_tensor = await convertDataUrlsToTensor([person_data_url!]);
        let person_inputs = {
          person: person_tensor as tf.Tensor4D,
        };
        person_graph_output = await this.person_graph(person_inputs);
        await this.person_graph_output_map.setNameTensorMap(
          person_key,
          person_graph_output
        );
      }
      let cloth_graph_output =
        await this.cloth_graph_output_map.getNamedTensorMap(cloth_key);
      if (cloth_graph_output === null) {
        let cloths_tensor = await convertImageUrlToTensor([cloth_path]);
        let cloths_mask_tensor = await convertMaskUrlToTensor([
          cloth_mask_path,
        ]);
        cloth_graph_output = {
          cloth_mask: cloths_mask_tensor,
          cloth: cloths_tensor,
        };
        await this.cloth_graph_output_map.setNameTensorMap(
          cloth_key,
          cloth_graph_output
        );
      }

      tryon_graph_output = await this.tryon_graph(
        cloth_graph_output,
        person_graph_output
      );
      await this.tryon_graph_output_map.setNameTensorMap(
        tryon_key,
        tryon_graph_output
      );

      tf.dispose(cloth_graph_output);
      tf.dispose(person_graph_output);

      let tryon_person_data_array = await converTensorToDataUrls(
        tryon_graph_output["person"] as tf.Tensor4D
      );
      tf.dispose(tryon_graph_output);
      return tryon_person_data_array;
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
    return this.person_graph_output_map.getExistingKeys();
  }
}
