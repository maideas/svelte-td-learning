<script>
  import { onMount } from "svelte";
  import { createEventDispatcher } from "svelte";
  import Data from "./Data.svelte";

  const dispatch = createEventDispatcher();

  export let numX;
  export let numY;
  export let numA;
  export let duelingQNet = false;
  export let learningRate = 0.002; // usually not needed to be overidden
  export let trainDistance = 10;

  const numStates = numX * numY;
  const maxData = 2000;
  const batchSize = 100;

  let trainDistanceCount = 0;
  let model = undefined;
  let DataComp;

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

  class DuelingLayer extends tf.layers.Layer {
    constructor() {
      super({});
    }
    getClassName() {
      return "DuelingLayer";
    }
    computeOutputShape(inputShape) {
      return inputShape[0];
    }
    call(input, kwargs) {
      return tf.tidy(() => {
        const A = input[0];
        const V = input[1];
        const axis = 1;
        return A.sub(A.mean(axis).reshape([-1, 1])).add(V);
      });
    }
  }

  const createDuelingModel = () => {
    const input = tf.input({ shape: [2] });

    const dense1 = tf.layers
      .dense({ units: 10, useBias: true, activation: "tanh" })
      .apply(input);
    const dense2 = tf.layers
      .dense({ units: 10, useBias: true, activation: "tanh" })
      .apply(dense1);

    const adv1 = tf.layers
      .dense({ units: 10, useBias: true, activation: "tanh" })
      .apply(dense2);
    const adv2 = tf.layers
      .dense({ units: numA, useBias: true, activation: "linear" })
      .apply(adv1);

    const val1 = tf.layers
      .dense({ units: 10, useBias: true, activation: "tanh" })
      .apply(dense2);
    const val2 = tf.layers
      .dense({ units: 1, useBias: true, activation: "linear" })
      .apply(val1);

    const output = new DuelingLayer().apply([adv2, val2]);

    model = tf.model({ inputs: input, outputs: output });
  };

  //====================================================

  const initModel = () => {
    if (model != undefined) {
      model.dispose();
    }

    if (duelingQNet) {
      createDuelingModel();
    } else {
      createModel();
    }

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

  export const updateModel = async (stepData, calcQValueFunc) => {
    // normalize state value and add it to stepData item ...
    stepData.normState = normalizeState(stepData.state);

    // add the given data item to the DataComp memory ...
    DataComp.add(stepData);

    // train Q network only every "trainDistance" steps ...
    trainDistanceCount++;
    if (trainDistanceCount < trainDistance) return;
    trainDistanceCount = 0;

    // get a random batch of data from the DataComp memory ...
    const stepDataBatch = DataComp.getBatch(batchSize);

    // prepare Q network input data (normalized state values) ...
    let normStates = [];
    stepDataBatch.forEach(stepData => {
      normStates.push(stepData.normState);
    });

    // get current Q network output data (Q values) for the
    // given input data (states) ...
    let QValues = predictNormStates(normStates);

    stepDataBatch.forEach((stepData, i) => {
      // update the selected actions related Q values ...
      QValues[i][stepData.a] = calcQValueFunc(stepData);
    });

    // use the prepared X and Y data to adjust the Q network ...
    await fit(normStates, QValues);

    // notify the upper components, that the model has changed ...
    dispatch("modelChanged", {});

    // tensorflow memory footprint debugging ...
    //console.table(tf.memory());
  };
</script>

<Data {maxData} bind:this={DataComp} />
