<script>
  import { onMount } from "svelte";

  export let title = "";
  export let xIsLog = false;

  export let data = [];
  export let yIsLog = false;
  export let yTitle = "";

  export let hasSecondY = false;
  export let dataSecond = [];
  export let ySecondIsLog = false;
  export let ySecondTitle = "";

  let plotDiv;

  export const updatePlot = () => {
    let xmin = data.length ? data.length * -0.02 : -0.02;
    let xmax = data.length ? data.length * 1.02 : 1.02;

    let layout = {
      title,
      showlegend: false,
      xaxis: {
        type: xIsLog ? "log" : "linear",
        range: [xmin, xmax]
      },
      yaxis: {
        title: yTitle,
        type: yIsLog ? "log" : "linear",
        range: [-1.1, 1.1],
        titlefont: { color: "#08C" },
        tickfont: { color: "#08C" },
        autorange: true
      },
      margin: {
        autoexpand: false,
        t: 50,
        l: 40,
        b: 30,
        r: 20
      }
    };
    if (yTitle != "") {
      layout.margin.l = 60;
    }

    let trace = {
      x: [...Array(data.length).keys()],
      y: data,
      mode: "lines",
      line: { shape: "spline" },
      type: "scatter"
    };
    let traces = [trace];

    if (hasSecondY) {
      layout.yaxis2 = {
        title: ySecondTitle,
        type: ySecondIsLog ? "log" : "linear",
        range: [-1.1, 1.1],
        overlaying: "y",
        side: "right",
        titlefont: { color: "#E60" },
        tickfont: { color: "#E60" },
        autorange: true
      };
      if (ySecondTitle != "") {
        layout.margin.r = 60;
      } else {
        layout.margin.r = 40;
      }

      let trace2 = {
        x: [...Array(dataSecond.length).keys()],
        y: dataSecond,
        mode: "lines",
        line: { shape: "spline" },
        type: "scatter",
        yaxis: "y2"
      };
      traces.push(trace2);
    }

    Plotly.react(plotDiv, traces, layout, { displaylogo: false });
  };

  export const clearPlot = () => {
    data = [];
    dataSecond = [];
    updatePlot();
  };

  onMount(() => {
    updatePlot();
  });
</script>

<style>
  div.plot {
    /* PlotlyJS requires content-box */
    -webkit-box-sizing: content-box;
    -moz-box-sizing: content-box;
    box-sizing: content-box;

    border-bottom: 1px solid #eee;
    width: 100%;
  }
</style>

<div class="plot" bind:this={plotDiv} />
