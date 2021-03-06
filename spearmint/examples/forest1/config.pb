language: PYTHON
name:     "forest"

variable {
 name: "lrate"
 type: FLOAT
 size: 1
 min:  1e-5
 max:  1
}

variable {
 name: "l2_reg"
 type: FLOAT
 size: 1
 min:  0
 max:  1
}

variable {
 name: "n_epochs"
 type: INT
 size: 1
 min:  5
 max:  2000

}

