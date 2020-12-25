<script>
  import { onMount } from "svelte";

  export let title = "";
  export let xIsLog = false;

  export let data = Array();
  export let yIsLog = false;
  export let yTitle = "";

  export let hasSecondY = false;
  export let dataSecond = Array();
  export let ySecondIsLog = false;
  export let ySecondTitle = "";

  let plotDiv;

  export const updatePlot = () => {
    let layout = {
      title,
      showlegend: false,
      xaxis: {
        type: xIsLog ? "log" : "linear",
        autorange: true
      },
      yaxis: {
        title: yTitle,
        type: yIsLog ? "log" : "linear",
        titlefont:  { color: "#08C" },
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
        overlaying: "y",
        side: "right",
        titlefont:  { color: "#E60" },
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
    data = Array();
    dataSecond = Array();
    updatePlot();
  };

  onMount(() => {
    updatePlot();
  });
</script>

<style>
  div.plot {
    border: 1px solid #eee;
    width: 100%;
  }
</style>

<div class="plot" bind:this={plotDiv} />
