package main

import (
	"io/ioutil"
	"log"
	"os"

	"github.com/spf13/pflag"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// RNNs require the context, so the beginning and the ending of each phrase cannot be processed.
// We prepend and append some silly text to overcome that.
const (
	header = "Meditation. I Am Happy, I Am Good, I Am Happy, I Am Good."
	footer = header
)

var CHARS = map[rune]uint8{
	'\n':     0,
	' ':      1,
	'e':      2,
	't':      3,
	'o':      4,
	'i':      5,
	'a':      6,
	's':      7,
	'n':      8,
	'r':      9,
	'l':      10,
	'h':      11,
	'u':      12,
	'd':      13,
	'c':      14,
	'm':      15,
	'.':      16,
	'p':      17,
	'y':      18,
	'f':      19,
	'g':      20,
	'w':      21,
	'b':      22,
	'v':      23,
	'k':      24,
	',':      25,
	'I':      26,
	':':      27,
	'\x02':   28,
	'\x03':   29,
	')':      30,
	'(':      31,
	'T':      32,
	'x':      33,
	'S':      34,
	'\'':     35,
	'A':      36,
	'/':      37,
	'-':      38,
	'=':      39,
	'C':      40,
	'1':      41,
	'j':      42,
	'"':      43,
	'E':      44,
	';':      45,
	'D':      46,
	'P':      47,
	'M':      48,
	'0':      49,
	'2':      50,
	'O':      51,
	'L':      52,
	'F':      53,
	'N':      54,
	'H':      55,
	'R':      56,
	'B':      57,
	'?':      58,
	'q':      59,
	'W':      60,
	'{':      61,
	'}':      62,
	'_':      63,
	'U':      64,
	'3':      65,
	'>':      66,
	'Y':      67,
	'<':      68,
	'4':      69,
	'5':      70,
	'[':      71,
	']':      72,
	'G':      73,
	'z':      74,
	'V':      75,
	'8':      76,
	'6':      77,
	'+':      78,
	'7':      79,
	'#':      80,
	'9':      81,
	'$':      82,
	'J':      83,
	'!':      84,
	'*':      85,
	'Q':      86,
	'X':      87,
	'@':      88,
	'K':      89,
	'%':      90,
	'\\':     91,
	'&':      92,
	'|':      93,
	'Z':      94,
	'^':      95,
	'~':      96,
	'`':      97,
	'’':      98,
	'”':      99,
	'“':      100,
	'…':      101,
	'—':      102,
	'–':      103,
	'\t':     104,
	'‘':      105,
	'\u200b': 106,
	'´':      107,
	'é':      108,
	'á':      109,
	'ü':      110,
	'ö':      111,
	'ä':      112,
	'ó':      113,
	'§':      114,
	'ç':      115,
	'à':      116,
	'í':      117,
	'£':      118,
	'©':      119,
	'»':      120,
	'�':      121,
	'€':      122,
	'×':      123,
	'→':      124,
	'а':      125,
	'°':      126,
	'è':      127,
	'и':      128,
	'о':      129,
	'е':      130,
	'ã':      131,
	'р':      132,
	'å':      133,
	'т':      134,
	'﹕':      135,
	'•':      136,
	'с':      137,
	'н':      138,
	'─':      139,
	'«':      140,
	'к':      141,
	'ñ':      142,
	'â':      143,
	'в':      144,
	'└':      145,
	'ú':      146,
	'л':      147,
	'Ã':      148,
	'ê':      149,
	'├':      150,
	'ا':      151,
	'д':      152,
	'®':      153,
	'у':      154,
	'ı':      155,
	'ø':      156,
	'™':      157,
	'¿':      158,
	'\u200c': 159,
	'м':      160,
	'│':      161,
	'ï':      162,
	'ß':      163,
	'，':      164,
	'✓':      165,
	'п':      166,
	'я':      167,
	'æ':      168,
	'²':      169,
	'ر':      170,
	'µ':      171,
	'ل':      172,
	'م':      173,
	'Ö':      174,
	'Ä':      175,
	'：':      176,
	'ë':      177,
	'·':      178,
	'Ü':      179,
	'б':      180,
	'Â':      181,
	'š':      182,
	'з':      183,
	'−':      184,
	'É':      185,
	'й':      186,
	'ы':      187,
	'ب':      188,
	'ł':      189,
	'ь':      190,
	'ô':      191,
	'г':      192,
	'±':      193,
	'ت':      194,
	'ن':      195,
	'Å':      196,
	'λ':      197,
	'و':      198,
}

func setup() []byte {
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "3")
	var modelPath string
	pflag.StringVarP(
		&modelPath, "model", "m", "", "Path to the trained model in Tensorflow GraphDef format.")
	pflag.Parse()
	if modelPath == "" {
		log.Fatal("-m / --model is required")
	}
	fi, err := os.Stat(modelPath)
	if err != nil {
		log.Fatalf("%s: %v", modelPath, err)
	}
	if fi.IsDir() {
		log.Fatalf("%s must be a file", modelPath)
	}
	data, err := ioutil.ReadFile(modelPath)
	if err != nil {
		log.Fatalf("Reading %s: %v", modelPath, err)
	}
	return data
}

func readInput(batchSize, sequenceLength int) ([]rune, []*tf.Tensor, []*tf.Tensor) {
	textBytes, err := ioutil.ReadAll(os.Stdin)
	if err != nil {
		log.Fatalf("Reading from stdin: %v", err)
	}
	if len(textBytes) == 0 {
		log.Fatal("empty input")
	}
	text := []rune(string(textBytes))
	realSize := len(text)
	inputSize := realSize
	if inputSize % batchSize != 0 {
		inputSize = (inputSize / batchSize + 1) * batchSize
	}
	batches1 := make([][][]uint8, inputSize / batchSize)
	batches2 := make([][][]uint8, inputSize / batchSize)
	for i := range batches1 {
		batches1[i] = make([][]uint8, batchSize)
		batches2[i] = make([][]uint8, batchSize)
	}
	for i := realSize; i < inputSize; i++ {
		arr := make([]uint8, sequenceLength)
		batches1[i / batchSize][i % batchSize] = arr
		batches2[i / batchSize][i % batchSize] = arr
	}
	augmentedText := []rune(header + string(text) + footer)
	pos := 0
	for x := range augmentedText {
		if x < len(header) {
			continue
		}
		if x >= len(header) + realSize {
			break
		}
		arr1 := make([]uint8, sequenceLength)
		batches1[pos / batchSize][pos % batchSize] = arr1
		arr2 := make([]uint8, sequenceLength)
		batches2[pos / batchSize][pos % batchSize] = arr2
		for i := 0; i < sequenceLength; i++ {
			bi := x - sequenceLength + i + 1
			if bi >= 0 {
				val, exists := CHARS[augmentedText[bi]]
				if !exists {
					val = uint8(len(CHARS))
				}
				arr1[i] = val
			}
			bi = x + sequenceLength - i
			if bi < len(augmentedText) {
				val, exists := CHARS[augmentedText[bi]]
				if !exists {
					val = uint8(len(CHARS))
				}
				arr2[i] = val
			}
		}
		pos++
	}
	tensors1 := make([]*tf.Tensor, len(batches1))
	tensors2 := make([]*tf.Tensor, len(batches2))
	for i := range batches1 {
		batch1 := batches1[i]
		batch2 := batches2[i]
		input1, err := tf.NewTensor(batch1)
		if err != nil {
			log.Fatalf("Creating the input tensor 1: %v", err)
		}
		input2, err := tf.NewTensor(batch2)
		if err != nil {
			log.Fatalf("Creating the input tensor 1: %v", err)
		}
		tensors1[i] = input1
		tensors2[i] = input2
	}
	return text, tensors1, tensors2
}

func main() {
	graph := tf.NewGraph()
	graph.Import(setup(), "")
	input1 := graph.Operation("input_1_1").Output(0)
	input2 := graph.Operation("input_2_1").Output(0)
	output := graph.Operation("output").Output(0)
	inputShape, err := input1.Shape().ToSlice()
	if err != nil {
		log.Fatalf("Getting the input shape: %v", err)
	}
	batchSize := int(inputShape[0])
	sequenceLength := int(inputShape[1])
	text, tensors1, tensors2 := readInput(batchSize, sequenceLength)
	session, err := tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		log.Fatalf("Failed to create a Tensorflow session: %v", err)
	}
	pos := 0
	for i := range tensors1 {
		result, err := session.Run(map[tf.Output]*tf.Tensor{
			input1: tensors1[i], input2: tensors2[i],
		}, []tf.Output{output}, nil)
		if err != nil {
			log.Fatalf("Failed to run Tensorflow session: %v", err)
		}
		probs := result[0].Value().([][]float32)
		for _, prob := range probs {
			print(string(text[pos]))
			maxi := 2
			maxval := prob[2]
			if prob[0] > maxval {
				maxi = 0
				maxval = prob[0]
			}
			if prob[1] > maxval {
				maxi = 1
				maxval = prob[1]
			}
			if maxi == 0 && pos < len(text) - 1 {
				print("<code>")
			}
			if maxi == 1 {
				print("</code>")
			}
			pos++
			if pos >= len(text) {
				break
			}
		}
	}
}
