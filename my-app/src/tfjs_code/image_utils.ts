export async function downloadImages(
  image_urls: string[]
): Promise<HTMLImageElement[]> {
  return await Promise.all(
    image_urls.map(async (image_url) => {
      let image = new Image();
      image.crossOrigin = "anonymous";
      let image_promise = onload2promise(image);
      image.src = image_url;
      await image_promise;
      return image;
    })
  );
}
export function isSupportedImageType(fileType: string): boolean {
  let set = new Set(["image/png", "image/jpeg", "image/heic"]);
  return set.has(fileType);
}

export async function convert_files_to_img_data(file: File): Promise<string> {
  let fileType = file.type;
  if (isSupportedImageType(fileType)) {
    let fileReader = new FileReader();
    let fileReaderPromise = onload2promise(fileReader);
    fileReader.readAsDataURL(file);
    await fileReaderPromise;
    return fileReader.result as string;
  } else {
    return Promise.reject(file.type);
  }
}

interface OnLoadAble {
  onload: any;
}
export function onload2promise<T extends OnLoadAble>(obj: T): Promise<T> {
  return new Promise((resolve, reject) => {
    obj.onload = () => resolve(obj);
  });
}
