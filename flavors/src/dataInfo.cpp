#include "dataInfo.h"
#include "utils.h"

#include "../json/json.hpp"

namespace Flavors
{
	void to_json(nlohmann::json& j, const DataInfo& info)
	{
		j["n"] = info.N;
		j["min"] = info.Min;
		j["max"] = info.Max;
		j["mean"] = info.Mean;
		j["variance"] = info.Variance();
		j["std"] = std::sqrt(info.VarianceN());
		j["skewness"] = info.Skewness();
		j["kurtosis"] = info.Kurtosis();
	}
}