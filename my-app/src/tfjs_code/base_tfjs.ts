import * as tf from "@tensorflow/tfjs-core";
import * as tfc from "@tensorflow/tfjs-converter";
import "@tensorflow/tfjs-backend-webgl";

import {
  convertMaskUrlToTensor,
  convertImageUrlsToTensor,
  converTensorToDataUrls,
  convertDataUrlsToTensor,
  downloadNameTensorMap,
} from "./core";
import { Tensor_Storage_Map } from "./tensor_storage_map";
import { EvictionPolicy, StorageType } from "./storage_map";

// copied naming patterns from tfjs
export type NamedModelMap = {
  [name: string]: tfc.GraphModel;
};

export type NamedTensor4DMap = {
  [name: string]: tf.Tensor4D;
};

export type NamedModelPathMap = {
  [name: string]: string;
};

export enum Mode {
  Regular,
  Debug, // downloads lots of outputs
  Express, // just returns input image; useful so when developing don't have to wait for models
}

export type ClothandMaskPath = [string, string];
export abstract class BaseTfjs {
  batchSize: number;
  modelsMap: Map<string, tfc.GraphModel> | undefined;
  modelsPresentIndexdbSet: Set<string>;
  modelsReady: boolean;
  mode: Mode;
  /**
   * This person graph output map is first in first out because we want to store most recent person images.
   *  It is used so that if you reload page don't have to upload person
   */
  personGraphOutputMap: Tensor_Storage_Map;
  /**
   *  The tryon output is first in last out because the idea if they reload the page, it will be scrolled back
   * to item 0. The tryon graph output is cached is used so that when you reload the page, we
   * can prepopulate the tryon image if possible.
   */
  tryonGraphOutputMap: Tensor_Storage_Map;
  abstract getModelsPathDict(): NamedModelPathMap;

  // dummy  abstract because can't declare abstract async methods
  async tryon_graph(
    cloth_graph_outputs: NamedTensor4DMap,
    person_graph_outputs: NamedTensor4DMap
  ): Promise<NamedTensor4DMap> {
    return { "not implemented": tf.tensor(0.0) };
  }
  // dummy  abstract because can't declare abstract async methods
  async person_graph(person: NamedTensor4DMap): Promise<NamedTensor4DMap> {
    return { "not implemented": tf.tensor(0.0) };
  }

  // should never have to change this but just in case we change inputs cloth_graph gives
  get_cloth_graph_dummy_outputs(): NamedTensor4DMap {
    return {
      cloth_mask: tf.ones([this.batchSize, 256, 192, 1]),
      cloth: tf.ones([this.batchSize, 256, 192, 3]),
    };
  }

  // should never have to change this but just in case input person ie shape
  get_person_input_dummy(): NamedTensor4DMap {
    return { person: tf.ones([5, 256, 192, 3]) };
  }

  /**
   *
   *
   *
   */
  constructor(batchSize: number, mode: Mode) {
    this.batchSize = batchSize;
    this.mode = mode;
    this.modelsPresentIndexdbSet = new Set<string>();
    this.modelsReady = false;

    this.personGraphOutputMap = new Tensor_Storage_Map(
      "person_graph_output_map",
      3,
      EvictionPolicy.FIRST_IN_FIRST_OUT,
      StorageType.INDEX_DB
    );
    this.tryonGraphOutputMap = new Tensor_Storage_Map(
      "tryon_graph_output_map",
      3,
      EvictionPolicy.FIRST_IN_LAST_OUT,
      StorageType.INDEX_DB
    );
    if (this.mode !== Mode.Express) {
      this.initializeProcess();
    }
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
              this.modelsPresentIndexdbSet.add(model_name);
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
    this.modelsMap = new Map(models_entries);
    this.modelsReady = true;
  }
  download_models_to_index_db() {
    this.modelsMap!.forEach((model, model_name) => {
      if (!this.modelsPresentIndexdbSet.has(model_name)) {
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
    console.log("we are in run model with dummy inputs" + tf.memory().numBytes);
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
    console.log("we are done running dumm inputs" + tf.memory().numBytes);
  }

  async runTryon(
    clothsAndMasksPath: ClothandMaskPath[],
    personKey: string,
    personDataUrl?: string
  ): Promise<string[]> {
    let tryonOutputs = await Promise.all(
      clothsAndMasksPath.map(async (clothAndMaskPath) => {
        let clothPath = clothAndMaskPath[0];
        let tryonKey = clothPath + "|" + personKey;
        return this.tryonGraphOutputMap.getNamedTensorMap(tryonKey);
      })
    );
    let uncachedClothsAndMasksPath = [];
    for (let i = 0; i < tryonOutputs.length; i++) {
      if (tryonOutputs[i] !== null) {
        uncachedClothsAndMasksPath.push(clothsAndMasksPath[i]);
      }
    }

    if (uncachedClothsAndMasksPath.length > 0) {
      let personGraphOutput = await this.personGraphOutputMap.getNamedTensorMap(
        personKey
      );
      if (this.mode === Mode.Express) {
        let personTensor;
        if (personGraphOutput === null) {
          personTensor = await convertDataUrlsToTensor([personDataUrl!]);
          await this.personGraphOutputMap.setNameTensorMap(personKey, {
            person: personTensor,
          });
        } else {
          personTensor = personGraphOutput["person"];
        }
        let resizedPerson = tf.image.resizeBilinear(personTensor, [256, 192]);
        let duplicateResizedPerson = tf.tile(resizedPerson, [
          clothsAndMasksPath.length,
          1,
          1,
          1,
        ]);
        let url = converTensorToDataUrls(duplicateResizedPerson);
        tf.dispose(personTensor);
        tf.dispose(duplicateResizedPerson);
        return url;
      }
      await this.ensureChecks();

      if (personGraphOutput === null) {
        if (personDataUrl) {
          let personTensor = await convertDataUrlsToTensor([personDataUrl]);
          let personInputs = {
            person: personTensor as tf.Tensor4D,
          };
          personGraphOutput = await this.person_graph(personInputs);
          await this.personGraphOutputMap.setNameTensorMap(
            personKey,
            personGraphOutput
          );
          tf.dispose(personInputs);
        } else {
          return Promise.reject("Person key does not exist");
        }
      }

      let uncachedClothsTensor = uncachedClothsAndMasksPath.map(
        (value) => value[0]
      );
      let uncachedMasksTensor = uncachedClothsAndMasksPath.map(
        (value) => value[1]
      );

      let clothsTensor = await convertImageUrlsToTensor(uncachedClothsTensor);
      let clothsMaskTensor = await convertMaskUrlToTensor(uncachedMasksTensor);
      let clothGraphOutput: NamedTensor4DMap = {
        cloth_mask: clothsMaskTensor,
        cloth: clothsTensor,
      };

      let unachedTryonGraphOutput = await this.tryon_graph(
        clothGraphOutput,
        personGraphOutput
      );

      if (this.mode === Mode.Debug) {
        await downloadNameTensorMap(clothGraphOutput);
        await downloadNameTensorMap(personGraphOutput);
        await downloadNameTensorMap(tryonGraphOutput);
      }
      tf.dispose(clothsTensor);
      tf.dispose(clothsMaskTensor);
      tf.dispose(clothGraphOutput);
      tf.dispose(personGraphOutput);
    }
    let tryonPersonDataArray = await converTensorToDataUrls(
      tryonGraphOutput["person"] as tf.Tensor4D
    );

    // We don't do wait for async because there is no gurantee to caller about this
    // and we want to do this relatively quickly
    this.tryonGraphOutputMap
      .setNameTensorMap(tryonKey, tryonGraphOutput)
      .then(() => tf.dispose(tryonGraphOutput!));

    return tryonPersonDataArray;
  }

  disposeModelFromGpu(): void {
    if (this.modelsMap) {
      this.modelsMap.forEach((model: tfc.GraphModel, _) => {
        model.dispose();
      });
    }
  }

  async ensureChecks(): Promise<void> {
    while (!this.modelsReady) {
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    return;
  }

  async getPersonKeys(): Promise<string[]> {
    return this.personGraphOutputMap.getExistingKeys();
  }
}
