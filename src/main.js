import MazeShell from './MazeShell.svelte';

const comp0Element = document.getElementById("maze-shell-0");
const comp1Element = document.getElementById("maze-shell-1");
const comp2Element = document.getElementById("maze-shell-2");
const comp3Element = document.getElementById("maze-shell-3");
const comp4Element = document.getElementById("maze-shell-4");

export const comp0 = comp0Element ? new MazeShell({
	target: comp0Element,
	props: {
		numX: 12,
		numY: 3,
		planningSteps: 10,  // Dyna-Q parameter

		blocked: Array(),
		terminal: Array(
			[1, 2], [2, 2], [3, 2],
			[4, 2], [5, 2], [6, 2],
			[7, 2], [8, 2], [9, 2],
			[10, 2],
			[11, 2]
		),
		rewards: Array(
			[1, 2, -10.0], [2, 2, -10.0], [3, 2, -10.0],
			[4, 2, -10.0], [5, 2, -10.0], [6, 2, -10.0],
			[7, 2, -10.0], [8, 2, -10.0], [9, 2, -10.0],
			[10, 2, -10.0],
			[11, 2, 0.0]
		),
		defaultReward: -0.1,
		startState: [0, 2]
	}
}) : null;

export const comp1 = comp1Element ? new MazeShell({
	target: comp1Element,
	props: {
		numX: 7,
		numY: 3,
		planningSteps: 10,  // Dyna-Q parameter

		blocked: Array(),
		terminal: Array(
			[1, 2], [2, 2], [3, 2], [4, 2], [5, 2],
			[6, 2]
		),
		rewards: Array(
			[1, 2, -10.0], [2, 2, -10.0], [3, 2, -10.0], [4, 2, -10.0], [5, 2, -10.0],
			[6, 2, 0.0]
		),
		defaultReward: -0.1,
		startState: [0, 2]
	}
}) : null;

export const comp2 = comp2Element ? new MazeShell({
	target: comp2Element,
	props: {
		numX: 5,
		numY: 5,
		blocked: Array(),
		terminal: Array([1, 1]),
		rewards: Array([1, 1, 1.0]),
		defaultReward: -0.1
	}
}) : null;

export const comp3 = comp3Element ? new MazeShell({
	target: comp3Element,
	props: {
		numX: 5,
		numY: 5,
		blocked: Array(),
		terminal: Array([1, 1]),
		rewards: Array([1, 1, 1.0], [3, 3, 1.0]),
		defaultReward: -0.1
	}
}) : null;

export const comp4 = comp4Element ? new MazeShell({
	target: comp4Element,
	props: {
		numX: 6,
		numY: 4,
		blocked: Array([0, 0], [3, 2], [4, 1], [3, 3]),
		terminal: Array([1, 1]),
		rewards: Array([1, 1, 10.0]),
		defaultReward: -0.1
	}
}) : null;
