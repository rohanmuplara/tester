export enum EvictionPolicy {
  FIRST_IN_FIRST_OUT,
  FIRST_IN_LAST_OUT,
}

/**
 * This writes and retrives values from local storage. Two main wrapper
 * functionalities it provides is that a. it provides a namespacing functionality
 * b. has some evictionPolicy c. and stores a list of keys currently in the table and this list
 * is persistent(it is also stored in localstorage.) The reason is if you reload the page, you know what keys are currently
 * stored.We store keys in fifo order.
 * d. converts all objects to json. However, the caller is responsible to make sure everything converts to json.
 */
export class Storage_Map {
  existing_keys: string[];
  namespace: string;
  maxItems: number;
  evictionPolicy: EvictionPolicy;
  constructor(
    namespace: string,
    maxItems: number,
    evictionPolicy: EvictionPolicy
  ) {
    this.namespace = namespace;
    let metaTablePath = namespace + ":metadata";
    let metaTableData = localStorage.getItem(metaTablePath);
    if (metaTableData === null) {
      this.existing_keys = [];
    } else {
      this.existing_keys = JSON.parse(metaTableData);
    }
    this.maxItems = maxItems;
    this.evictionPolicy = evictionPolicy;
  }
  setItem(key: string, value: any): void {
    let namespacedKey = this.namespace + ":" + key;
    let jsonValue = JSON.stringify(value);
    localStorage.setItem(namespacedKey, jsonValue);
    if (this.existing_keys.length === this.maxItems) {
      let removal_key;
      if (this.evictionPolicy === EvictionPolicy.FIRST_IN_FIRST_OUT) {
        removal_key = this.existing_keys.shift();
      } else if (this.evictionPolicy === EvictionPolicy.FIRST_IN_LAST_OUT) {
        removal_key = this.existing_keys.pop();
      }
      let removal_namespaced_key = this.namespace + ":" + removal_key;
      localStorage.removeItem(removal_namespaced_key);
    }
    this.existing_keys.push(key);
    let metaTablePath = this.namespace + ":metadata";
    localStorage.setItem(metaTablePath, JSON.stringify(this.existing_keys));
  }
  getItem(key: string): any {
    let namespaced_key = this.namespace + ":" + key;
    let json_value = localStorage.getItem(namespaced_key);
    if (json_value) {
      return JSON.parse(json_value!);
    } else {
      return null;
    }
  }

  getExistingKeys(): string[] {
    return this.existing_keys;
  }
}
