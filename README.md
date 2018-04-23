## Animation Pin Tool

The purpose of this tool is you *pin* your controls in place while you move dependent controls/nodes on the rig.
Ie, pin the chest while you move the hips. 

Or better yet, pin **all** the IK controls while you move the base node (the one that sets the character's x, y, z). 

Right now, the UI is being overhauled, so all you get are two buttons. 

![Imgur](https://i.imgur.com/ImuKZqe.png)

It will default the bake range to be your timeline edit range (the inner values). If you're saavy, you can feed the create function a start/end frame to mess with it. Just check it out.

When you bake the pins, it will remember the start/end frame from the creation procedure. Unless you want it to only bake a certain region within that - you could highlight part of your timeline and use it that way as well.



The bake option defaults to baking back down to your previous keys. Right now it remembers the keys on rotates and translates separately (as an experiment) and will clean the bake using those as a reference.

I think I might need to default to compositing translate and rotate keys to maintain accuracy. But for now, give it a shot and see if it's useful.

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

The pull up a quick UI, Make this a <b>python</b> button on your shelf. 

```python
# Python command to show the UI
import animPin as animPin
animPin.show()
```

To execute the methods without a UI, use these python commands:

Create:
```python
import animPin as animPin
animPin.create_pins() # To create the pins based off your selection
```

Bake:
```python
animPin.bake_pins(bake_option = 1) # To bake the selected pin groups
```

**[Warnings](#warnings)**

Ok, this tool is untested. I mean, I've tested it at home, but not out 
in the wild.

The biggest note I would say is to *not* use it with _animation layers_!