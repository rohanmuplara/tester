import { NamedTensorMap } from "@tensorflow/tfjs-core";
import * as tf from "@tensorflow/tfjs-core";
import { Storage_Map } from "./storage_map";
import { convertDataUrlsToTensor, converTensorToDataUrls } from "./core";

export class Tensor_Storage_Map extends Storage_Map {
  async setNameTensorMap(key: string, namedTensorMap: NamedTensorMap) {
    let nameArrayMap = await Promise.all(
      Object.entries(namedTensorMap).map(async ([key, tensor]) => {
        let dataUrls = await converTensorToDataUrls(tensor as tf.Tensor4D);
        let lastDimensionShape = tensor.shape[3];
        return [
          key,
          { lastDimensionShape: lastDimensionShape, dataUrls: dataUrls },
        ];
      })
    ).then(Object.fromEntries);
    this.setItem(key, nameArrayMap);
  }
  async getNamedTensorMap(key: string): Promise<NamedTensorMap | null> {
    let nameValueMap = await this.getItem(key);
    if (nameValueMap) {
      let nameTensorEntries = await Promise.all(
        Object.entries(nameValueMap).map(
          async ([key, serializedDict]: [string, any]) => {
            //let tensor = await convertDataUrlsToTensor(dataUrls as string[]);
            let lastDimensionShape = serializedDict["lastDimensionShape"];
            let dataUrls = serializedDict["dataUrls"];
            let tensor = await convertDataUrlsToTensor(dataUrls);
            let newTensor;
            if (lastDimensionShape === 1) {
              newTensor = tf.split(tensor, 3, 3)[0];
              tf.dispose(tensor);
            } else {
              newTensor = tensor;
            }
            return [key, newTensor];
          }
        )
      );
      return Object.fromEntries(nameTensorEntries) as NamedTensorMap;
    } else {
      return null;
    }
  }
}
