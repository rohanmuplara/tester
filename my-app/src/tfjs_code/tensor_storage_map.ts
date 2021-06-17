import { NamedTensorMap } from "@tensorflow/tfjs";
import * as tf from "@tensorflow/tfjs";
import { Storage_Map } from "./storage_map";

export class Tensor_Storage_Map extends Storage_Map {
  async setNameTensorMap(key: string, namedTensorMap: NamedTensorMap) {
    let nameArrayMap = await Promise.all(
      Object.entries(namedTensorMap).map(async ([key, tensor]) => {
        let tensor_array = await tensor.array();
        return [key, tensor_array];
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
        ([key, array]) => {
          let tensor = tf.tensor(array as number[]);
          return [key, tensor];
        }
      );
      return Object.fromEntries(nameTensorEntries) as NamedTensorMap;
    } else {
      return null;
    }
  }
}
