import { NamedTensorMap } from "@tensorflow/tfjs-core";
import * as tf from "@tensorflow/tfjs-core";
import { Storage_Map } from "./storage_map";

export class Tensor_Storage_Map extends Storage_Map {
  async setNameTensorMap(key: string, namedTensorMap: NamedTensorMap) {
    let nameArrayMap = await Promise.all(
      Object.entries(namedTensorMap).map(async ([key, tensor]) => {
        //let data_urls = await converTensorToDataUrls(tensor as tf.Tensor4D);
        return [key, tensor.arraySync()];
      })
    ).then(Object.fromEntries);
    this.setItem(key, nameArrayMap);
  }
  async getNamedTensorMap(key: string): Promise<NamedTensorMap | null> {
    let nameArrayMap = await this.getItem(key);
    if (nameArrayMap) {
      let nameTensorEntries = await Promise.all(
        Object.entries(nameArrayMap).map(async ([key, dataUrls]) => {
          //let tensor = await convertDataUrlsToTensor(dataUrls as string[]);
          let tensor = tf.tensor(dataUrls as number[]);
          return [key, tensor];
        })
      );
      return Object.fromEntries(nameTensorEntries) as NamedTensorMap;
    } else {
      return null;
    }
  }
}
