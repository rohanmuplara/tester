import localForage from "localforage";

export enum EvictionPolicy {
  FIRST_IN_FIRST_OUT,
  FIRST_IN_LAST_OUT,
}

export enum StorageType {
  LOCALSTORAGE,
  INDEX_DB,
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
  readonly existingKeyPath = "___existingKeys___";
  existingKeys: string[];
  maxItems: number;
  evictionPolicy: EvictionPolicy;
  storageInstance: LocalForage;
  initializationFinished: boolean;

  constructor(
    namespace: string,
    maxItems: number,
    evictionPolicy: EvictionPolicy,
    storage: StorageType,
  ) {
    this.existingKeys = [];
    this.initializationFinished = false;
    this.storageInstance = localForage.createInstance({
      name: namespace,
      driver:
        storage == StorageType.INDEX_DB
          ? localForage.INDEXEDDB
          : localForage.LOCALSTORAGE,
    });
    this.storageInstance
      .getItem(this.existingKeyPath)
      .then((existingKeys: any) => {
        this.existingKeys = existingKeys !== null ? existingKeys : [];
        this.initializationFinished = true;
      });
    this.maxItems = maxItems;
    this.evictionPolicy = evictionPolicy;
  }
  async setItem(key: string, value: any): Promise<void> {
    await this.ensureChecks()
    await this.storageInstance.setItem(key, value);
    if (this.existingKeys.length === this.maxItems) {
      let removalKey: string;
      if (this.evictionPolicy === EvictionPolicy.FIRST_IN_FIRST_OUT) {
        removalKey = this.existingKeys.shift()!;
      } else if (this.evictionPolicy === EvictionPolicy.FIRST_IN_LAST_OUT) {
        removalKey = this.existingKeys.pop()!;
      }
      await this.storageInstance.removeItem(removalKey!);
    }
    this.existingKeys.push(key);
    await this.storageInstance.setItem(this.existingKeyPath, this.existingKeys);
  }
  async getItem(key: string): Promise<any> {
    await this.ensureChecks()
    return this.storageInstance.getItem(key);
  }

  async getExistingKeys(): Promise<string[]> {
    await this.ensureChecks();
    return this.existingKeys;
  }

  async ensureChecks(): Promise<void> {
    while (!this.initializationFinished) {
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
    return;
  }
  
}
