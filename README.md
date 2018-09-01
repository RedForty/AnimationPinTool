## Animation Pin Tool

<img align="left" style="float: left; padding-right: 20px" src="https://i.imgur.com/CBqXf93.png">
The purpose of this tool is you *pin* your controls in place while you move dependent controls/nodes on the rig.
Ie, pin the chest while you move the hips. 

Or better yet, pin **all** the IK controls while you move the base node (the one that sets the character's x, y, z). 

You can get crazier and even reframe the animation of your character relative to your moving camera. Experiment. Go crazy!

The UI has a right-click menu to be able to miniaturize it. RMB anywhere to access the option.


It will default the bake range to be your timeline edit range (the inner values). 

When you bake the pins, it will remember the start/end frame from the creation procedure. Unless you want it to only bake a certain region within that - you could highlight select part of your timeline and use it that way as well.

The bake option defaults to baking back down to your previous keys. Right now it remembers the keys on rotates and translates and composites the new poses on those frames by keying all transforms together. IE, it will flatten sparse rotates and translates. By default, match keys will remember the empty space between your keys and delete the extra keys created by the baking process.

A lot of it is in flux, but the tool does what I mean for it to do. If you have any ideas, let me know. For now, give it a shot and see if it's useful.



---
**[Installation](#installation)**

Place both of these files in your maya/scripts folder. 
The other files do not matter.

```
~maya/scripts/
            - animPin.py
            - Qt.py
```

- Qt is a project by Marcus Ottosson-> [https://github.com/mottosso/Qt.py](https://github.com/mottosso/Qt.py)

---

**[Usage](#usage)**

To pull up a quick UI, Make this a <b>python</b> button on your shelf. 

```python
# Python command to show the UI
import animPin as animPin
animPin.show()
```

To execute the methods without a UI, use these python commands:

If you're saavy and you want to use the python commands, you can feed the create function a start/end frame to mess with it. Start by feeding it the variables, start_frame = # and end_frame = #. Check it out.

Create:
```python
import animPin as animPin
animPin.create_pins(start_frame = 0, end_frame = 420)
```

Bake:
```python
animPin.bake_pins(bake_option = 0) # To bake the selected pin groups
```

---

**[Warnings](#warnings)**

Ok, this tool is untested. I mean, I've tested it at home, but not out 
in the wild.

The biggest note I would say is to *not* use it with _animation layers_!
