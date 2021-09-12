<script>
  import { onMount } from "svelte";
  import { createEventDispatcher } from "svelte";
  import Data from "./Data.svelte";

  const dispatch = createEventDispatcher();

  export let numX;
  export let numY;
  export let numA;
  export let learningRate = 0.002; // usually not needed to be overidden

  const numStates = numX * numY;
  const maxData = 2000;
  const batchSize = 100;

  let model = undefined;

  let DataComp; // represents the experience replay buffer

  //====================================================

  const createModel = () => {
    model = tf.sequential();

    model.add(
      tf.layers.dense({
        inputShape: [2],
        units: 10,
        useBias: true,
        activation: "tanh"
      })
    );
    model.add(
      tf.layers.dense({ units: 10, useBias: true, activation: "tanh" })
    );
    model.add(
      tf.layers.dense({ units: 10, useBias: true, activation: "tanh" })
    );
    model.add(
      tf.layers.dense({ units: numA, useBias: true, activation: "linear" })
    );
  };

  //====================================================

  const initModel = () => {
    if (model != undefined) {
      model.dispose();
    }

    createModel();

    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: "meanSquaredError",
      metrics: ["accuracy"]
    });
  };

  //====================================================

  const fit = async (dataX, dataY) => {
    const epochs = 10;

    const trainX = tf.tensor2d(
      dataX,
      [dataX.length, dataX[0].length],
      "float32"
    );
    const trainY = tf.tensor2d(
      dataY,
      [dataY.length, dataY[0].length],
      "float32"
    );

    const onEpochEnd = (n, logs) => {
      console.log("onEpochEnd", "[ epoch", n, "of", epochs, "]", logs);
    };

    await model.fit(trainX, trainY, {
      batchSize: dataX.length,
      epochs: epochs,
      shuffle: true,
      callbacks: {
        /* onEpochEnd */
      }
    });

    trainX.dispose();
    trainY.dispose();
  };

  //====================================================

  const normalize = (dataX, min, max) => {
    let dataN = [];
    dataX.forEach((x, i) => {
      x -= (max[i] + min[i]) / 2.0;
      x /= (max[i] - min[i]) / 2.0;
      dataN.push(x);
    });
    return dataN;
  };

  const normalizeState = state => {
    return normalize(state, [0, 0], [numX - 1, numY - 1]);
  };

  //====================================================
  // init
  //====================================================

  export const init = () => {
    initModel();
    DataComp.clear();

    // notify the upper components, that the model has changed ...
    dispatch("modelChanged", {});
  };

  onMount(() => {
    init();
  });

  //====================================================
  // predict
  //====================================================

  const predictNormStates = dataX => {
    let a;
    tf.tidy(() => {
      const X = tf.tensor2d(dataX, [dataX.length, dataX[0].length], "float32");
      a = model.predict(X).arraySync();
    });
    return a;
  };

  export const predictState = state => {
    return predictNormStates([normalizeState(state)])[0];
  };

  //====================================================
  // update
  //====================================================

  export const updateModel = (stepData, calcLogitsFunc) => {
    // normalize state value ...
    let normState = normalizeState(stepData.state);

    // get current network output data (logits) for the
    // given input data (normalized state) ...
    let logits = predictNormStates([normState])[0];

    // get updated logits values ...
    logits = calcLogitsFunc(logits, stepData);

    // add the data item to the DataComp memory ...
    DataComp.add({ normState, logits });
  };

  export const takeModel = async () => {
    for (let n = 0; n < Math.ceil(DataComp.length() / batchSize); n++) {
      // get a random batch of data from the DataComp memory ...
      const stepDataBatch = DataComp.getBatch(batchSize);

      // prepare logits network train data ...
      let normStates = [];
      let logits = [];
      stepDataBatch.forEach(stepData => {
        normStates.push(stepData.normState);
        logits.push(stepData.logits);
      });

      // use the prepared X and Y data to adjust the logits network ...
      await fit(normStates, logits);
    }

    // now, we clear the episode data to prevent the usage of outdated
    // trajectory data at following model fit calls ...
    DataComp.clear();

    // notify the upper components, that the model has changed ...
    dispatch("modelChanged", {});

    // tensorflow memory footprint debugging ...
    //console.table(tf.memory());
  };
</script>

<Data {maxData} bind:this={DataComp} />
