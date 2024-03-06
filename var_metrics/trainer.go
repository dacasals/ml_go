package var_metrics

import "fmt"

func Trainer(path string) {

	POP := []string{"YES", "NO", "YES", "YES", "NO", "MAYBE"}
	TROLL2 := []string{"NO", "YES", "MAYBE", "NO", "YES", "NO"}
	metric := MutualInformation{}

	mi := metric.miDiscrete(TROLL2, POP)
	fmt.Printf("Mutual Information %g", mi)

}
