# Console Interface
The console interface enables interaction with Python scripts through the command line.

Launch via `gui='console'` in [`view()`](phi/vis/index.html#phi.vis.view) or [`show()`](phi/vis/index.html#phi.vis.show).

The console interface runs on a different thread than the main Python script.

## Commands

Multiple commands can be chained using `;` to separate them.

Actions are registered as new commands.
E.g. if an action  `my_function()` is

### General Commands

`help` Prints available commands.

`status` Prints the current status.

`controls` Prints a list of available controls and their values.

`actions` Prints a list of available actions.

### Plotting

`show` Plots the first two fields.

`show <fields>`

`show <field>.<component>` where `<component>` must be one of `x`, `y`, `z`, `abs`

### Control Execution

`play` Runs all loop iterations until paused.

`play <frames>` Runs a certain number of iterations. Has no effect if already playing.

`pause` Pauses loop execution after the current iteration finishes.

`step` Progresses the loop by one iteration.

`<control_name> = <value>` Sets a control value.

`exit` Finishes the current loop iteration and exits the program.

`kill` Immediately stops the Python process.

