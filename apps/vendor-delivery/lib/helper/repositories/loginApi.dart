import 'package:egrocer_seller/helper/utils/generalImports.dart';

Future<Map<String, dynamic>> getLoginRepository(
    {required Map<String, dynamic> params}) async {
  var response = await GeneralMethods.sendApiRequest(
      apiName: ApiAndParams.apiLogin, params: params, isPost: true);

  return json.decode(response);
}
