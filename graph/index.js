
const tryon = async (setPrediction) => {

    // https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/blazeface/tfjs/model.json.gz
    const grapyModel = `https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/grapy/atr_512_256_mobilenet_edge_loss_1/tfjs/model.json.gz`;
    const segModel = `https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/tops/expseg/expected_seg_debug/tfjs/model.json.gz`;
    const tpsModel = `https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/tops/tps/short_tshirts_default/tfjs/model.json.gz`;
    const tomModel = `https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/tops/tom/model_60/tfjs/model.json.gz`;
    const clothModel = `https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/tops/model_800/tfjs/model.json.gz`;
    const blazeModel = `https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/blazeface/tfjs/model.json.gz`;
    const denseposeModel = `https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/densepose/densepose_js/model.json.gz`;

    const gridImageUrl = 'https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/images/grid_img.png';
    const gradientImageUrl = 'https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/images/gradient_img.png';
   const personImageUrl = 'https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/images/person_img.png';
    const clothImageUrl = 'https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/images/cloth_img.png';


    function filterMask(mask, valList) {
        let newmask = tf.cast(tf.equal(mask, valList[0]), 'float32');
        valList.slice(1).forEach(val => {
            let temp = tf.cast(tf.equal(mask, val), 'float32');
            newmask = newmask.add(temp);
        });
        return newmask
    }

    function getClothOutput(modelpath, cloth_input) {
       console.log('cloth output starting')
       console.time('cloth model load');
        return (tf.loadGraphModel(modelpath))
            .then(model => {
                console.timeEnd('cloth model load');
                return tf.tidy(() => {

                    console.time('cloth output');
                    const predictions = model.predict(cloth_input);
                    console.timeEnd('cloth output');
                    return predictions;
                })
            })
    }

    function getBlazefaceOutput(modelpath, bf_input) {
        // console.log('----------')
       console.log('blaze output starting')
       console.time('blaze model load');

        return (tf.loadGraphModel(modelpath))
            .then(model => {
               console.timeEnd('blaze model load');

                return tf.tidy(() => {
                             
                    console.time('blaze output');
                    const predictions = model.predict(bf_input);
                    console.timeEnd('blaze output');
                    return predictions;
                })
            })
    }

    function getGrapyOutput(modelpath, grapy_input) {
        console.time("grapy model");
        return (tf.loadGraphModel(modelpath))
            .then(model => {
                console.timeEnd("grapy model");
                return tf.tidy(() => {

                    console.time("grapy output");
                    const predictions = model.predict(grapy_input);
                    console.timeEnd("grapy output");
                    return predictions;
                })
            })
    }

    function getDenseposeOutput(modelpath, dp_input) {
        console.time("denspose model");

        return (tf.loadGraphModel(modelpath))
            .then(model => {
              console.timeEnd("denspose model");


                return tf.tidy(() => {
                    console.time("denspose output");

                    const predictions = model.predict(dp_input);
                    console.timeEnd("denspose output");

                    return predictions;
                })
            })
    }

    function getExpsegOutput(modelpath, seg_inputs) {
       console.time("exp model");
        return (tf.loadGraphModel(modelpath))
            .then(model => {
                console.timeEnd("exp model");

                return tf.tidy(() => {

                    console.time("exp output");
                    const predictions = model.predict(seg_inputs);
                    console.timeEnd("exp output");
                    return predictions;
                })
            })
    }

    function getTpsOutput(modelpath, tps_inputs) {
        console.time("tps model");
        return (tf.loadGraphModel(modelpath))
            .then(model => {
                console.timeEnd("tps model");
                console.time("tps output");
                return tf.tidy(() => {

                    const predictions = model.predict(tps_inputs);
                    console.timeEnd("tps output");
                    return predictions;
                })
            })
    }

    function getTomOutput(modelpath, tom_inputs) {
        console.time("tom model");
        return (tf.loadGraphModel(modelpath))
            .then(model => {
                console.timeEnd("tom model");

                console.time("tom output");
                return tf.tidy(() => {
                    const predictions = model.predict(tom_inputs);
                    console.timeEnd("tom output");
                    return predictions;
                })
            })
    }


    function resize(img, size, mask = false) {
        let frame_h = size[0]
        let frame_w = size[1]
        // console.log(img.shape, img.shape.slice([1], [3]))
        let image_h = img.shape.slice([1], [3])[0]
        let image_w = img.shape.slice([1], [3])[1]

        // console.log(frame_h, frame_w, image_h, image_w)
        let h_r = frame_h / image_h
        let w_r = frame_w / image_w
        let r = Math.min(h_r, w_r)
        let dim = [Math.round(image_h * r), Math.round(image_w * r)]
        // console.log("in resize: ", r, dim, h_r, w_r)

        let resized;
        if (mask) {
            // console.log("if");

            resized = tf.image.resizeNearestNeighbor(img, [dim[0], dim[1]], false, true); // Removed alignCorners and halfPixelCenters
        } else {
            // console.log("else2");
            resized = tf.image.resizeBilinear(img, [dim[0], dim[1]], true, false); // Removed alignCorners and halfPixelCenters
        }

        let delta_h = frame_h - resized.shape[1];
        let delta_w = frame_w - resized.shape[2];

        let top = Math.floor(delta_h / 2);
        let bottom = delta_h - Math.floor(delta_h / 2);
        let left = Math.floor(delta_w / 2);
        let right = delta_w - Math.floor(delta_w / 2);
        // console.log(img.shape);

        // console.log(image_h);
        // console.log(image_w);

        // console.log(top);
        // console.log(bottom);
        // console.log(left);
        // console.log(right);
        // console.log(resized.shape);

        let new_im = resized.pad([
            [0, 0],
            [top, bottom],
            [left, right],
            [0, 0]
        ]);

        // let z = tf.zeros([800,600,1])
        // resized = tf.concat([resized, z], 2)
        // resized = tf.util.flatten(resized.arraySync())

        return [new_im, [top, bottom, left, right]];

    }

    function restore_resize(img, size, paddings, mask = false) {
        let top = paddings[0]
        let bottom = paddings[1]
        let left = paddings[2]
        let right = paddings[3]
        // console.log("paddings: ", paddings, size)
        // console.log("before", img.shape)
        img = img.slice([0, top, left, 0], [img.shape[0], (img.shape[1] - bottom - top), (img.shape[2] - right - left), img.shape[3]])
        // console.log("after", img.shape)


        if (mask) {
            // console.log('--- 1')
            img = tf.image.resizeNearestNeighbor(img, size, false, true); // Removed alignCorners and halfPixelCenters
        } else {
            // console.log('--- 2')
            img = tf.image.resizeBilinear(img, size, true, false); // Removed alignCorners and halfPixelCenters
        }
        // console.log(img.shape)
        return img
    }

    function transform_center_to_corner(boxes) {
        let c1 = tf.sub(boxes.slice([0, 0], [boxes.shape[0], 2]), tf.div(boxes.slice([0, 2], [boxes.shape[0], 2]), 2));
        let c2 = tf.add(boxes.slice([0, 0], [boxes.shape[0], 2]), tf.div(boxes.slice([0, 2], [boxes.shape[0], 2]), 2));
        let corner_box = tf.concat([
            c1,
            c2
        ], -1); // Removed axis
        return corner_box;
    }

    function decode(default_boxes, locs, variance = [0.1, 0.2]) {
        // console.log("locs.shape: ", locs.shape)
        // console.log("default_boxes.shape: ", default_boxes.shape)
        let x0 = default_boxes.slice([0, 0], [default_boxes.shape[0], 2]);
        let x = default_boxes.slice([0, 2], [default_boxes.shape[0], 2]);
        let a0 = tf.fill(locs.slice([0, 0], [locs.shape[0], 2]).shape, variance[0]);
        let a1 = tf.fill(locs.slice([0, 2], [locs.shape[0], 2]).shape, variance[1]);
        let y = locs.slice([0, 2], [locs.shape[0], 2]);
        let y0 = locs.slice([0, 0], [locs.shape[0], 2])
        // console.log("test: ", tf.exp(tf.mul(y, a1)).shape, x.shape, tf.mul(tf.exp(tf.mul(y, a1)), x).shape)
        let c1 = tf.add(tf.mul(tf.mul(y0, a0), x), x0);
        let c2 = tf.mul(tf.exp(tf.mul(y, a1)), x);
        locs = tf.concat([
            c1,
            c2
        ], -1); // Removed axis
        let boxes = transform_center_to_corner(locs);
        return boxes;
    }
    async function predict_bbox(predictions, anchors) {
        let conf_score = predictions.slice([0, 0, 4], [predictions.shape[0], predictions.shape[1], 1])
        let loc_score = predictions.slice([0, 0, 0], [predictions.shape[0], predictions.shape[1], 4])
        let confs = conf_score.squeeze(0)
        let locs = loc_score.squeeze(0)
        let pred_boxes = decode(anchors, locs)
        // console.log("pred_boxes.shape: ", pred_boxes.shape)

        let cls_scores = confs.slice([0, 0], [confs.shape[0], 1])
        // console.log("cls_scores.shape: ", cls_scores.shape)

        let score_idx = tf.greater(cls_scores, 0)
        // console.log("score_idx.shape: ", score_idx.shape)

        let cls_boxes = await tf.booleanMaskAsync(pred_boxes, score_idx.squeeze(-1))
        cls_scores = await tf.booleanMaskAsync(cls_scores, score_idx)
        let cls_labels = 1 * cls_boxes.shape[0]
        let boxes = tf.clipByValue(cls_boxes, 0.0, 1.0).arraySync()
        let score = tf.max(cls_scores)
        let index = await tf.whereAsync(tf.equal(cls_scores, score))
        // console.log("tf.gatherND(boxes, index): ", tf.gatherND(boxes, index).arraySync()[0])
        xmin = tf.gatherND(boxes, index).arraySync()[0][0]
        ymin = tf.gatherND(boxes, index).arraySync()[0][1]
        xmax = tf.gatherND(boxes, index).arraySync()[0][2]
        ymax = tf.gatherND(boxes, index).arraySync()[0][3]
        // console.log("xmin, ymin,  xmax, ymax: ", xmin, ymin, xmax, ymax)
        return [xmin, ymin, xmax, ymax]
    }

    function generate_anchors() {
        let anchor_specs = [
            [16, [0.43],
                [0.63, 0.43]
            ],
            [8, [0.50, 0.30],
                [0.43, 0.49, 0.63]
            ]
        ]
        let anchors = [];
        anchor_specs.forEach((item) => {
            let feature_map_size = item[0]
            let scale_list = item[1];
            let ratio_list = item[2]
            for (let x = 0; x < feature_map_size; x++) {
                // console.log(x)
                for (let y = 0; y < feature_map_size; y++) {
                    let x_center = (x + 0.5) / feature_map_size
                    let y_center = (y + 0.5) / feature_map_size
                    ratio_list.forEach((ratio) => {
                        scale_list.forEach((scale) => {
                            let w = scale * ratio
                            let h = scale / ratio
                            // console.log([x_center, y_center, w, h])
                            anchors.push([x_center, y_center, w, h])
                        })
                    })
                }
            }
        })
        // console.log(anchors.length)
        anchors = tf.tensor(anchors)
        anchors = tf.clipByValue(anchors, 0.0, 1.0)
        return anchors;
    }

    // person = tf.tensor(person)

    // grid_img = tf.tensor(grid_img)
    // gradient_image = tf.tensor(gradient_image)
    // cloth_inp = tf.tensor(cloth_inp)

    async function getTensorFromImageUrl(imageUrl) {
        let img;
        const imageLoadPromise = new Promise(resolve => {
            img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = resolve;
            img.src = imageUrl;
        });
    
        await imageLoadPromise;
        // console.log("image loaded");
        return tf.cast(tf.browser.fromPixels(img).expandDims(0), 'float32');
    }

    const saveAsJson = async (myData, fileName) => {

        // const fileName = fileName;
        const DEV = false;

        if(DEV){
            const json = JSON.stringify(myData);
            const blob = new Blob([json],{type:'application/json'});
            const href = await URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = href;
            link.download = fileName + ".json";
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
      }


    let grid_img = await getTensorFromImageUrl(gridImageUrl);
    let gradient_image = await getTensorFromImageUrl(gradientImageUrl);
    let person = await getTensorFromImageUrl(personImageUrl);
    let cloth_inp = await getTensorFromImageUrl(clothImageUrl);


    // Normalize
    cloth_inp = cloth_inp.div(127.5);
    cloth_inp = cloth_inp.sub(1);

    person = person.div(127.5);
    person = person.sub(1);

    // grid_img = (grid_img / 255.0 - [0.5, 0.5, 0.5]) / [0.5, 0.5, 0.5]
    grid_img = grid_img.div(255.0)
    grid_img = grid_img.sub([0.5, 0.5, 0.5])
    grid_img = grid_img.div([0.5, 0.5, 0.5])

    gradient_image = gradient_image.div(255.0)
    gradient_image = gradient_image.sub([0.5, 0.5, 0.5])
    gradient_image = gradient_image.div([0.5, 0.5, 0.5])


    // console.log("shape: ", person.shape, cloth_inp.shape);
    saveAsJson(person.arraySync(), 'person')
    saveAsJson(cloth_inp.arraySync(), 'cloth');
    saveAsJson(person.arraySync(), 'grid_img')
    saveAsJson(cloth_inp.arraySync(), 'gradient_image');


    // console.log(grid_img);

    let xmin = 47;
    let ymin = 0;
    let xmax = 138;
    let ymax = 235;


    // var canvas = document.createElement("canvas");
    // document.body.appendChild(canvas);



    let cloth_input = resize(cloth_inp, [256, 192])[0]
    saveAsJson(cloth_input.arraySync(), 'cloth_resized');

    let cloth_paddings = resize(cloth_inp, [256, 192])[1]

    var clothOutputPromise = getClothOutput(clothModel, cloth_input);
    clothOutputPromise.then(async (cloth_out) => {
        // console.log("cloth_out: ", cloth_out)
        saveAsJson(cloth_out.arraySync(), 'cloth_output');

        let cloth_mask = tf.argMax(cloth_out, -1).expandDims(-1); // Removed axis
        cloth_mask = tf.cast(tf.greater(cloth_mask, 0.5), 'float32');

        let cloth = resize(cloth_inp, [256, 192])[0]
        cloth = tf.add(tf.mul(cloth, cloth_mask), tf.sub(1, cloth_mask))
        // console.log('SHAPES: ', cloth.shape)


        let bf_input = resize(person, [256, 256])[0]
        saveAsJson(bf_input.arraySync(), 'bf_input');

        let paddings = resize(person, [256, 256])[1]

        bf_input = tf.div(tf.mul(tf.add(bf_input, 1), 127.5), 255)

        // console.log("bf input: ", bf_input, paddings)

        var blazefaceOutputPromise = getBlazefaceOutput(blazeModel, bf_input);
        blazefaceOutputPromise.then(async (bf_out) => {
            saveAsJson(bf_out.arraySync(), 'bf_output');

            // console.log("bf_out: ", bf_out)
            let anchors = generate_anchors();
            // console.log("anchors: ", anchors);
            let boxes = await predict_bbox(bf_out, anchors)
            // console.log("boxes: ", boxes)
            let xmin = boxes[0] * 256;
            let ymin = boxes[1] * 256;
            let xmax = boxes[2] * 256;
            let ymax = boxes[3] * 256;
            let top = paddings[0];
            let bottom = paddings[1];
            let left = paddings[2];
            let right = paddings[3];

            let w_r = person.shape[2] / bf_input.shape[2]
            let h_r = person.shape[1] / bf_input.shape[1]

            xmin = xmin - Math.floor(left / 2);
            ymin = ymin - Math.floor(top / 2);
            xmax = xmax + Math.floor(right / 2);
            ymax = ymax + Math.floor(bottom / 2);

            xmin = parseInt(xmin * w_r)
            ymin = parseInt(ymin * h_r)
            xmax = parseInt(xmax * w_r)
            ymax = parseInt(ymax * h_r)
            console.log("Blazeface xmin, ymin, xmax, ymax: ", xmin, ymin, xmax, ymax)


            let cropped_person = person.slice([0, ymin, xmin, 0], [person.shape[0], ymax - ymin, xmax - xmin, person.shape[3]])
            let input = cropped_person;
            saveAsJson(cropped_person.arraySync(), 'grapy_cropped_input');

            let grapy_input = resize(input, [512, 256])[0]
            paddings = resize(input, [512, 256])[1];

            saveAsJson(grapy_input.arraySync(), 'grapy_input');

            var grapyOutputPromise = getGrapyOutput(grapyModel, grapy_input);
            grapyOutputPromise.then(async (grapy_out) => {
                // console.log("grapy_out: ", grapy_out)
                saveAsJson(grapy_out.arraySync(), 'grapy_output');

                grapy_out = tf.argMax(grapy_out, -1).expandDims(3); // Removed axis
                grapy_out = tf.cast(grapy_out, 'float32')

                // console.log("grapy_out: ", grapy_out)
                grapy_out = restore_resize(grapy_out, [input.shape[1], input.shape[2]], paddings, true) // Removed mask


                grapy_out = tf.pad(grapy_out, [
                    [0, 0],
                    [ymin, person.shape[1] - ymax],
                    [xmin, person.shape[2] - xmax],
                    [0, 0]
                ])
                // console.log("grapy_out: ", grapy_out, grapy_out.squeeze(0).squeeze(2))

                let indices = await tf.whereAsync(tf.greater(grapy_out.squeeze(0).squeeze(2), 0))
                // console.log(indices.shape)

                let col = indices.slice([0, 0], [indices.shape[0], 1])
                let row = indices.slice([0, 1], [indices.shape[0], 1])


                xmin = tf.min(row)
                ymin = tf.min(col)
                xmax = tf.max(row)
                ymax = tf.max(col)
                // console.log("xmin, ymin, xmax, ymax: ", xmin.dataSync(), ymin.dataSync(), xmax.dataSync(), ymax.dataSync())
                grapy_out = grapy_out.slice([0, ymin.dataSync()[0], xmin.dataSync()[0], 0], [grapy_out.shape[0], tf.sub(ymax, ymin).dataSync()[0], tf.sub(xmax, xmin).dataSync()[0], grapy_out.shape[3]])
                // console.log("grapy_out: ", grapy_out.shape)
                person = person.slice([0, ymin.dataSync()[0], xmin.dataSync()[0], 0], [person.shape[0], tf.sub(ymax, ymin).dataSync()[0], tf.sub(xmax, xmin).dataSync()[0], person.shape[3]])
                // console.log("person: ", person.shape)

                person = resize(person, [256, 192])[0]
                grapy_out = resize(grapy_out, [256, 192], true)[0] // Removed mask
                // console.log("grapy_out: ", grapy_out.shape)
                // console.log("person: ", person.shape)


                // **************** DENSEPOSE *********************
                // console.log("person.shape: ", person.shape)
                let person_mask = tf.cast(tf.greater(grapy_out, 0), 'float32')
                person = tf.add(tf.mul(person, person_mask), tf.sub(1, person_mask))
                let dp_input = person;
                saveAsJson(dp_input.arraySync(), 'dp_input');

                var denseposeOutputPromise = getDenseposeOutput(denseposeModel, dp_input);
                denseposeOutputPromise.then(async (dp_out) => {
                    saveAsJson(dp_out.arraySync(), 'dp_output');

                    // console.log(dp_out.shape)
                    let dp_seg = tf.cast(tf.argMax(dp_out.slice([0, 0, 0, 0], [dp_out.shape[0], dp_out.shape[1], dp_out.shape[2], 25]), -1), 'float32') // Removed axis
                    // console.log(dp_seg.shape)
                    dp_seg = tf.cast(dp_seg, 'int32')
                    // [dp_out.shape[0], dp_out.shape[1], dp_out.shape[2], dp_out.shape[3]]
                    let dp_uv = dp_out.slice([0, 0, 0, 25])
                    // console.log(dp_uv.shape)
                    dp_seg = tf.cast(tf.oneHot(dp_seg, 25), 'float32');
                    dp_out = tf.concat([dp_seg, dp_uv], -1); // Removed axis
                    // console.log(dp_out.shape)



                    // *********************** SEG ******************************************
                    let shape_mask = tf.cast(filterMask(grapy_out, [1, 3, 8, 9, 10, 11, 12, 13, 14]), 'float32')
                    let pants_mask = tf.cast(filterMask(grapy_out, [3]), 'float32')
                    let pants = tf.mul(person, pants_mask)

                    let seg_inputs = {
                        'input_1': shape_mask,
                        'input_2': dp_out,
                        'input_3': cloth,
                        'input_4': cloth_mask,
                        'input_5': pants
                    }

                    saveAsJson(shape_mask.arraySync(), 'es_shape_mask_input');
                    saveAsJson(dp_out.arraySync(), 'es_dp_output_input');
                    saveAsJson(cloth.arraySync(), 'es_cloth_input');
                    saveAsJson(cloth_mask.arraySync(), 'es_cloth_mask_input');
                    saveAsJson(pants.arraySync(), 'es_pants_input');


                    var expsegOutputPromise = getExpsegOutput(segModel, seg_inputs);
                    expsegOutputPromise.then(async (expseg) => {
                        // console.log("expseg: ", expseg, expseg.length)

                        let segmap = tf.argMax(expseg[1], -1).expandDims(-1)
                        // console.log("segmap.shape: ", segmap.shape)

                        let unoccluded_torso = tf.argMax(expseg[0], -1).expandDims(-1);
                        unoccluded_torso = tf.cast(unoccluded_torso, 'float32');

                        // ******************************* TPS ********************************
                        let person_cloth_mask = unoccluded_torso
                        let gt_warped_mask = person_cloth_mask
                        let gt_warped_cloth = tf.mul(person, gt_warped_mask)

                        // console.log("gt_warped_cloth shapes: ", cloth.shape, cloth_mask.shape, gt_warped_mask.shape, grid_img.shape, gradient_image.shape)
                        let tps_inputs = {
                            'input_2': cloth,
                            'input_3': cloth_mask,
                            'input_4': gt_warped_mask,
                            'input_5': grid_img,
                            'input_6': gradient_image,
                        }
                        saveAsJson(gt_warped_mask.arraySync(), 'gt_warped_mask');
                        saveAsJson(grid_img.arraySync(), 'grid_img');
                        saveAsJson(gradient_image.arraySync(), 'gradient_image');
        
        

                        var tpsOutputPromise = getTpsOutput(tpsModel, tps_inputs);
                        tpsOutputPromise.then(async (tps_out) => {
                            // console.log("tps_out: ", tps_out)
                            let warped_mask = tps_out[2]
                            let warped_cloth = tps_out[1]
                            // console.log(warped_mask.shape, warped_cloth.shape)
                            warped_cloth = tf.mul(warped_cloth, warped_mask)
                            // console.log(warped_cloth.shape)

                            saveAsJson(warped_cloth.arraySync(), 'tps_warped_cloth')
                            saveAsJson(warped_mask.arraySync(), 'tps_warped_mask')
                            // *************************** TOM ************************************
                            let pass_skin = tf.ones([1, 256, 192, 1], 'float32')
                            let hand_mask = tf.mul(tf.mul(filterMask(grapy_out, [6, 7]), filterMask(segmap, [4, 5])), pass_skin)

                            let person_priors_mask = tf.add(filterMask(segmap, [6]), hand_mask)
                            // console.log("shapes: ", pass_skin.shape, hand_mask.shape, person_priors_mask.shape, filterMask(segmap, [4, 5]).shape)
                            let person_priors = tf.mul(person, person_priors_mask)

                            let tom_inputs = {
                                'input_1': warped_cloth, // 3
                                'input_2': person_priors, // 3
                                'input_3': tf.cast(segmap, 'float32') // 1
                            }

                            saveAsJson(warped_cloth.arraySync(), 'tom_warped_cloth_input')
                            saveAsJson(person_priors.arraySync(), 'tom_person_priors_input')
                            saveAsJson(segmap.arraySync(), 'tom_segmap_input')


                            var tomOutputPromise = getTomOutput(tomModel, tom_inputs);
                            tomOutputPromise.then(async (tom_out) => {
                                console.log("tom_out: ", tom_out);
                                const output = tf.cast(tom_out[2].squeeze(0).add(1.).mul(127.5), 'int32');

                                // tf.browser.toPixels(tf.cast(output, 'int32'), canvas);
                                // return tf.cast(output, 'int32');
                                saveAsJson(tom_out[0].arraySync(), 'tom_out[0]')
                                saveAsJson(tom_out[1].arraySync(), 'tom_out[1]')
                                saveAsJson(tom_out[2].arraySync(), 'tom_out[2]')
                                saveAsJson(tom_out[3].arraySync(), 'tom_out[3]')
                                saveAsJson(tf.cast(output, 'int32').arraySync(), 'output')
                                let output_ds = output.arraySync();
                                let  cnt = 0
                                for(var i=0; i<output_ds.length; i++){
                                    for(var j=0; j<output_ds[0].length; j++){
                                        for(var k=0; k<output_ds[0][0].length; k++){
                                            cnt+=1;
                                            if(output_ds[i][j][k] < 0)output_ds[i][j][k]=0;
                                            if(output_ds[i][j][k] > 255)output_ds[i][j][k]=255;
                                        }
                                    }
                                }
                                console.timeEnd("everything");
                                console.log(cnt);

                            })
                        })
                    })

                })
            })

        });

    });

}
console.time("everything");
tryon();
