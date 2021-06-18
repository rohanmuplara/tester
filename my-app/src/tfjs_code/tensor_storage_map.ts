import { NamedTensorMap } from "@tensorflow/tfjs";
import * as tf from "@tensorflow/tfjs";
import { Storage_Map } from "./storage_map";
import { convertDataUrlsToTensor, converTensorToDataUrls } from "./core";

export class Tensor_Storage_Map extends Storage_Map {
  async setNameTensorMap(key: string, namedTensorMap: NamedTensorMap) {
    let nameArrayMap = await Promise.all(
      Object.entries(namedTensorMap).map(async ([key, tensor]) => {
        let data_urls = await converTensorToDataUrls(tensor as tf.Tensor4D);
        return [key, data_urls];
      })
    ).then(Object.fromEntries);
    let jsonStringifiedObject = JSON.stringify(nameArrayMap);
    this.setItem(key, jsonStringifiedObject);
  }
  async getNamedTensorMap(key: string): Promise<NamedTensorMap | null> {
    let namespaced_key = this.namespace + ":" + key;
    let json_value = await this.getItem(namespaced_key);
    if (json_value) {
      let nameArrayMap = JSON.parse(json_value);
      let nameTensorEntries = Object.entries(nameArrayMap).map(
        ([key, dataUrls]) => {
          let tensor = convertDataUrlsToTensor(dataUrls as string[]);
          return [key, tensor];
        }
      );
      return Object.fromEntries(nameTensorEntries) as NamedTensorMap;
    } else {
      return null;
    }
  }
}
