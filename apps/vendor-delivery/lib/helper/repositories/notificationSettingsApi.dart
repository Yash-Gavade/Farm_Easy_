import 'package:egrocer_seller/helper/utils/generalImports.dart';

Future<Map<String, dynamic>> getNotificationSettingsRepository(
    {required Map<String, dynamic> params}) async {
  var response = await GeneralMethods.sendApiRequest(
      apiName: ApiAndParams.apiNotificationSettings,
      params: params,
      isPost: false);

  return json.decode(response);
}

Future<Map<String, dynamic>> updateNotificationSettingsRepository(
    {required Map<String, dynamic> params}) async {
  var response = await GeneralMethods.sendApiRequest(
      apiName: ApiAndParams.apiNotificationSettingsUpdate,
      params: params,
      isPost: true);

  return json.decode(response);
}
