extends Node2D

# Called when the node enters the scene tree for the first time.
func _ready():
    var xi = [0, 1, 1, 1, 1, 1]
    var xj = [0, 0, 0, 0, 1, 1]
    var y = [0, 0, 0, 0, 1, 1]

    print("Running training and predict on existing node")

    $EditorKomputeModelMLNode.train(y, xi, xj)

    var preds = $EditorKomputeModelMLNode.predict(xi, xj)

    print(preds)

    print("Running training and predict on new instance")

    # Create new instance
    var s = KomputeModelMLNode.new()

    s.train(y, xi, xj)
    print("")

    preds = s.predict(xi, xj)

    print(preds)

