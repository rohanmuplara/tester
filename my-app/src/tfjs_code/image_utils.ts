import heic2any from "heic2any";

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

export function getKeyNameFromFile(file: File): string {
  return file.name + file.lastModified;
}

export async function convert_file_to_img_data(file: File): Promise<string> {
  let fileType = file.type;
  if (isSupportedImageType(fileType)) {
    let fileReader = new FileReader();
    let fileReaderPromise = onload2promise(fileReader);
    fileReader.readAsDataURL(file);
    await fileReaderPromise;
    let dataUrl = fileReader.result as string;
    if (fileType === "image/heic") {
      let fetch_result = await fetch(dataUrl);
      let heic_blob = (await fetch_result.blob()) as Blob;
      let pngBlob = await heic2any({ blob: heic_blob, toType: "image/png" });
      let pngDataUrl = URL.createObjectURL(pngBlob);
      return pngDataUrl;
    }
    return dataUrl;
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

export function downloadImageDataUrls(
  imageDataUrls: string[],
  names: string[]
) {
  imageDataUrls.forEach((imageDataUrl, index) => {
    let fake_link = document.createElement("a");
    console.log("entering fake link");
    fake_link.download = names[index];
    fake_link.href = imageDataUrl;
    fake_link.click();
  });
}
