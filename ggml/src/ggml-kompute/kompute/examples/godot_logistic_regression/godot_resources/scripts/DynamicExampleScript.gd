extends Node2D

onready var xi_node = $UI/UIVBoxContainer/XIHBoxContainer/LineEdit
onready var xj_node = $UI/UIVBoxContainer/XJHBoxContainer/LineEdit
onready var y_node = $UI/UIVBoxContainer/XJHBoxContainer/LineEdit
onready var preds_node = $UI/UIVBoxContainer/Panel/VBoxContainer/PredHBoxContainer2/PredictionsLabel
onready var w1_node = $UI/UIVBoxContainer/Panel/VBoxContainer/PredHBoxContainer/Weight1Label
onready var w2_node = $UI/UIVBoxContainer/Panel/VBoxContainer/PredHBoxContainer/Weight2Label
onready var bias_node = $UI/UIVBoxContainer/Panel/VBoxContainer/PredHBoxContainer/BiasLabel

# Called when the node enters the scene tree for the first time.
func _ready():
    pass

func compute_ml():

    var xi = str2var(xi_node.text)
    var xj = str2var(xj_node.text)
    var y = str2var(y_node.text)

    var s = KomputeModelML.new()

    s.train(y, xi, xj)

    var preds = s.predict(xi, xj)

    preds_node.text = str(preds)

    var params = s.get_params()

    w1_node.set_text(str(params[0]))
    w2_node.set_text(str(params[1]))
    bias_node.set_text(str(params[2]))



